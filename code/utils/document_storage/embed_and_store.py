"""
embed_and_store.py

Push per-page Markdown files for a single case directory into Pinecone.

*Default* = idempotent: a page whose **vector ID** already exists in the
index is skipped. Pass `overwrite=True` to force a re-embed (upsert will
replace any existing vector with the same ID).

Vector **ID design**
====================
    <case_id>|<doc_type>|<sha256(text)[:16]>

Metadata:
    {
      "vector_id":      "<the same ID as the Pinecone key>",
      "case_id":        "23CHLC22869",
      "doc_type":       "complaint",
      "page_file":      "page_1.md",
      "content_hash":   "<full_sha256>",
      "embedding_model":"text-embedding-3-small",
      "ingested":       "2025-06-17T19:59:00Z"
    }

Usage:
    python embed_and_store.py \
        --case-dir files/cases_parsed/23CHLC22869 \
        --index-name lasc \
        --overwrite
"""

import os
import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# ───────────────────────── Config Defaults ────────────────────────────────
load_dotenv()
DEFAULT_INDEX      = os.getenv("PINECONE_INDEX", "lasc")
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# Instantiate once
pc_client     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
default_embed  = OpenAIEmbeddings(model=DEFAULT_EMBED_MODEL)

# ───────────────────────── Helper Functions ───────────────────────────────

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_id(case_id: str, doc_type: str, text: str) -> str:
    return f"{case_id}|{doc_type}|{sha256(text)[:16]}"

# ───────────────────────── Core Routine ────────────────────────────────────
def identify_exhibits_fallback(doc_type: str) -> Dict[str, str]:
    md_file_to_exhibit_metadata = {}    
    pages = sorted(doc_type.glob("page_*.[mM][dD]"),    # grab every page_*.md
                   key=lambda x: int(x.stem.split('_')[1])   # sort by the numeric part
                  )
    for md_file in pages:
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # If we have not yet found an exhibit page separator, then check if the current page is an exhibit page separator
        if "plaintiff" in text.lower() or "defendant" in text.lower():
            # If the page mentions "plaintiff" or "defendant", we assume it's an allegation page
            exhibit_metadata = "allegation"
        else:
            exhibit_metadata = "exhibit"
        md_file_to_exhibit_metadata[md_file.name] = exhibit_metadata
    return md_file_to_exhibit_metadata

def identify_exhibits(doc_type: str) -> Dict[str, str]:
    """
    For each md file in the directory, determine whether the text is a page separator for exhibits.
    This is a heuristic based on the content of the page.
    Create a mapping from md filename to a boolean indicating whether it is an exhibit page.
    """
    md_file_to_exhibit_metadata = {}    
    pages = sorted(doc_type.glob("page_*.[mM][dD]"),    # grab every page_*.md
                   key=lambda x: int(x.stem.split('_')[1])   # sort by the numeric part
                  )
    pattern = re.compile(r'\bExhibit\s+(["\']?)([A-Za-z0-9]+)\1', re.IGNORECASE)
    exhibit_metadata = "allegation"
    for md_file in pages:
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # If we have not yet found an exhibit page separator, then check if the current page is an exhibit page separator
        if exhibit_metadata != "exhibit":
            # Use regex to check Exhibit in any caps, ignoring quotation marks, followed by any letter or number
            exhibit_match = pattern.search(text)
            # Check that the word count is low
            if exhibit_match and (len(text.split(' ')) < 20):
                # Found an exhibit page separator! Switch the metadata to "exhibit" for this and all subsequent pages
                # To obtain metadata regarding which page pertains to which exhibit (e.g., A, B, C), can use: str.upper(exhibit_match.group(2))
                exhibit_metadata = "exhibit"

        md_file_to_exhibit_metadata[md_file.name] = exhibit_metadata
    
    if exhibit_metadata != "exhibit":
        # If we never found an exhibit page separator, then resort to the back-up method of vocabulary
        md_file_to_exhibit_metadata = identify_exhibits_fallback(doc_type)
        
    return md_file_to_exhibit_metadata

def embed_and_store(
    case_dir: str,
    *,
    pinecone_index_name: str = DEFAULT_INDEX,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    overwrite: bool = False,
):
    """
    Embed every Markdown page under `case_dir` into Pinecone.

    Parameters
    ----------
    case_dir : str
        Path like `files/cases_parsed/<case_id>/`
    pinecone_index_name : str
        Name of the Pinecone index to upsert into.
    embedding_model : str
        Name of the OpenAI embedding model.
    overwrite : bool
        If False, skip vectors whose ID already exists.
        If True, upsert all pages, replacing existing vectors.
    """
    index   = pc_client.Index(pinecone_index_name)
    embedder = (
        default_embed
        if embedding_model == DEFAULT_EMBED_MODEL
        else OpenAIEmbeddings(model=embedding_model)
    )

    case_id = Path(case_dir).name
    pages: List[Dict] = []

    # 1) Collect pages
    for doc_type in sorted(Path(case_dir).iterdir()):
        if not doc_type.is_dir():
            continue
        
        md_file_to_exhibit_metadata = identify_exhibits(doc_type)

        for md_file in sorted(doc_type.glob("*.[mM][dD]")):
            text = md_file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            vid = _vector_id(case_id, doc_type.name, text)
            pages.append({
                "id": vid,
                "text": text,
                "meta": {
                    "text": text,            
                    "vector_id": vid,
                    "case_id":   case_id,
                    "doc_type":  doc_type.name,
                    "exhibit_or_allegation": md_file_to_exhibit_metadata[md_file.name],
                    "page_file": md_file.name,
                    "content_hash": sha256(text),
                    "embedding_model": embedding_model,
                    "ingested":  datetime.now(timezone.utc)
                                            .isoformat(timespec="seconds"),
                }
            })

    if not pages:
        print(f"[WARN] No pages found under {case_dir}")
        return

    # 2) Skip logic
    if not overwrite:
        existing = set(index.fetch(ids=[p["id"] for p in pages]).vectors.keys())
        pages = [p for p in pages if p["id"] not in existing]
        if not pages:
            print(f"[SKIP] All pages already indexed for case {case_id}")
            return

    # 3) Embed & upsert
    batch = []
    for p in tqdm(pages, desc="Embedding pages", unit="page"):
        vec = embedder.embed_query(p["text"])
        batch.append((p["id"], vec, p["meta"]))

    index.upsert(vectors=batch)
    mode = "overwrite" if overwrite else "new only"
    print(f"✅ Upserted {len(batch)} pages ({mode}) for case {case_id}")

# ───────────────────────── CLI Helper ──────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir",       required=True)
    parser.add_argument("--index-name",     default=DEFAULT_INDEX)
    parser.add_argument("--embedding-model",default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--overwrite",      action="store_true")
    args = parser.parse_args()

    embed_and_store(
        args.case_dir,
        pinecone_index_name=args.index_name,
        embedding_model=args.embedding_model,
        overwrite=args.overwrite,
    )
