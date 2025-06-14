import os
import re
import base64
from typing import Optional
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat

# ─── Setup Azure client once ──────────────────────────────────────────────
load_dotenv()
AZURE_ENDPOINT = os.getenv("azure_endpoint")
AZURE_KEY      = os.getenv("azure_key")

client = DocumentIntelligenceClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY)
)


def extract_case_number(text: str) -> Optional[str]:
    """Find first pattern like 23CHLC22869 in filename or path."""
    m = re.search(r"\d{2}[A-Z]{4,5}\d{5}", text)
    return m.group(0) if m else None


def classify_document(first_page_md: str) -> str:
    """Classify by page-1 content."""
    txt = first_page_md.lower()
    if "civ-105" in txt:
        return "request_for_default_judgment"
    if "pld-c-001" in txt or "complaint for" in txt:
        return "complaint"
    return "other"


def analyze_and_save_local_pdf(
    pdf_path: str,
    raw_dir: str = "files/raw_cases",
    out_dir: str = "files/cases_parsed",
):
    # 1) Case extraction
    case_id = extract_case_number(pdf_path)
    if not case_id:
        print(f"[SKIP] no case_id in {pdf_path}")
        return

    # 2) Read & encode bytes
    with open(pdf_path, "rb") as f:
        b = f.read()
    payload = base64.b64encode(b).decode("utf-8")

    # 3) Send to Azure
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body={"base64Source": payload},
        output_content_format=DocumentContentFormat.MARKDOWN,
    )
    result = poller.result()

    # 4) Split result.content into per-page Markdown
    page_texts = []
    for page in result.pages:
        span = page.spans[0]
        start, end = span.offset, span.offset + span.length
        page_texts.append(result.content[start:end])

    # 5) Classify based on first page
    doc_type = classify_document(page_texts[0])

    # 6) Write out under case/<doc_type>/
    final_folder = os.path.join(out_dir, case_id, doc_type)
    os.makedirs(final_folder, exist_ok=True)

    for idx, text in enumerate(page_texts, start=1):
        fn = os.path.join(final_folder, f"page_{idx}.md")
        with open(fn, "w", encoding="utf-8") as out:
            out.write(text)

    print(f"[DONE] {case_id} / {doc_type} → {len(page_texts)} pages.")


def batch_process_local_pdfs(
    raw_dir: str = "files/raw_cases",
    out_dir: str = "files/cases_parsed",
):
    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        analyze_and_save_local_pdf(
            pdf_path=os.path.join(raw_dir, fname),
            raw_dir=raw_dir,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    batch_process_local_pdfs()
