"""
pdf_to_markdown_pipeline.py

Convert court-filing PDFs into per-page Markdown via Azure Document Intelligence,
with Pinecone as a shared ledger so that no two runs re-OCR the same file.

Usage:
    python pdf_to_markdown_pipeline.py [--case-id CASE_ID] [--overwrite]

Examples:
    # Process all cases
    python pdf_to_markdown_pipeline.py

    # Only process a single docket
    python pdf_to_markdown_pipeline.py --case-id 23CHLC22869

    # Force re-OCR of everything for one case
    python pdf_to_markdown_pipeline.py --case-id 23CHLC22869 --overwrite
"""

import os
import re
import base64
import hashlib
import shutil
import time
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from pinecone import Pinecone

# ───────────────────────── Config Defaults ────────────────────────────────
RAW_DIR   = Path("files/raw_cases")
OUT_DIR   = Path("files/cases_parsed")
LEDGER_NS = "ocr_ledger"
# primary index holds both real vectors and the ledger namespace
DEFAULT_INDEX = os.getenv("PINECONE_INDEX", "lasctesttwo")

load_dotenv()

# Azure Document Intelligence client
ai_client = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_KEY")),
)

# ──────────────────────────── Main Pipeline ───────────────────────────────
def batch_run(
    *,
    case_id: Optional[str] = None,
    overwrite: bool = False,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = OUT_DIR,
    index_name: str = DEFAULT_INDEX,
    ledger_ns: str = LEDGER_NS,
):
    """
    Process PDFs in `raw_dir`, write per-page .md to `out_dir`, and skip
    already-processed files via a Pinecone ledger namespace in the same index.

    Parameters
    ----------
    case_id : Optional[str]
        If provided, only PDFs whose case number matches are processed.
    overwrite : bool
        If True, re-OCR even if the file was processed before.
    """
    pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index  = pc.Index(index_name)
    # get index dimension for padding dummy vectors
    idx_meta = pc.describe_index(index_name)
    dim = idx_meta.dimension

    def already_processed(pdf_hash: str) -> bool:
        return bool(
            index.fetch(ids=[pdf_hash], namespace=ledger_ns).vectors
        )

    def mark_processed(pdf_hash: str):
        # pad to index dimension with one non-zero at the start
        dummy = [1.0] + [0.0] * (dim-1)
        index.upsert(
            [(pdf_hash, dummy, {"ts": time.time()})],
            namespace=ledger_ns
        )

    def ocr_pdf(path: Path):
        data     = path.read_bytes()
        pdf_hash = hashlib.sha256(data).hexdigest()

        if not overwrite and already_processed(pdf_hash):
            print(f"skip {path.name}")
            return

        cid = case_number_from(path.name)
        if not cid:
            print(f"no case #: {path.name}")
            return
        if case_id and cid != case_id:
            return

        poller = ai_client.begin_analyze_document(
            model_id="prebuilt-layout",
            body={"base64Source": base64.b64encode(data).decode()},
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        res = poller.result()

        pages = [
            res.content[s.offset : s.offset + s.length]
            for page in res.pages for s in page.spans[:1]
        ]
        if not pages:
            print(f"0 pages: {path.name}")
            return

        doc_type = classify(pages[0])
        tgt_dir  = out_dir / cid / doc_type
        if overwrite:
            shutil.rmtree(tgt_dir, ignore_errors=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)

        for i, md in enumerate(pages, start=1):
            (tgt_dir / f"page_{i}.md").write_text(md, encoding="utf-8")

        mark_processed(pdf_hash)
        print(f"done {cid}/{doc_type}, pages={len(pages)}")

    for pdf in sorted(raw_dir.glob("*.pdf")):
        ocr_pdf(pdf)

# ──────────────────────────── Helper Functions ───────────────────────────
def case_number_from(filename: str) -> Optional[str]:
    match = re.search(r"\d{2}[A-Z]{4,5}\d{5}", filename)
    return match.group(0) if match else None

def classify(md_first_page: str) -> str:
    t = md_first_page.lower()
    if "civ-105" in t:
        return "request_for_default_judgment"
    if "pld-c-001" in t or "complaint for" in t:
        return "complaint"
    return "other"

# ──────────────────────────── CLI Entry ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", help="Only process this case ID")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    batch_run(case_id=args.case_id, overwrite=args.overwrite)
