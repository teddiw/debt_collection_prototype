import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings


def embed_ocr_pages_to_pinecone(
    case_dir: str,
    pinecone_index_name: str = "lasc",
    embedding_model: str = "text-embedding-3-small",
):
    """
    Walk all subfolders under `case_dir` (complaint, request_for_default_judgment, other),
    load each page_*.md, and upsert only the pages not yet in Pinecone.
    """
    load_dotenv()
    pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(pinecone_index_name)
    embed = OpenAIEmbeddings(model=embedding_model)

    case_id = Path(case_dir).name
    pages = []

    # 1) collect all pages with their metadata & deterministic IDs
    for doc_type in sorted(os.listdir(case_dir)):
        doc_dir = os.path.join(case_dir, doc_type)
        if not os.path.isdir(doc_dir):
            continue

        for fname in sorted(os.listdir(doc_dir)):
            if not fname.lower().endswith((".md", ".txt")):
                continue
            text = (Path(doc_dir) / fname).read_text(encoding="utf-8").strip()
            if not text:
                continue

            page_id = f"{case_id}_{doc_type}_{fname}"
            pages.append({
                "id":       page_id,
                "text":     text,
                "metadata": {"case_id": case_id, "doc_type": doc_type, "source": fname}
            })

    if not pages:
        print(f"[WARN] No pages found under {case_dir}")
        return

    # 2) check which IDs are already indexed
    all_ids     = [p["id"] for p in pages]
    resp        = index.fetch(ids=all_ids)
    existing_ids = set(resp.vectors.keys())

    # 3) filter out already‐indexed pages
    to_upsert = [p for p in pages if p["id"] not in existing_ids]
    skipped    = len(pages) - len(to_upsert)

    if not to_upsert:
        print(f"[SKIP] All {len(pages)} pages already in Pinecone for case {case_id}")
        return

    # 4) embed & upsert remaining pages
    upsert_batch = []
    for p in tqdm(to_upsert, desc="Embedding new pages"):
        vec = embed.embed_query(p["text"])
        upsert_batch.append((p["id"], vec, p["metadata"]))

    # Pinecone SDK expects a list of (id, vector, metadata)
    index.upsert(vectors=upsert_batch)

    print(
        f"✅ Indexed {len(to_upsert)} new pages "
        f"(skipped {skipped} existing) "
        f"from case {case_id}."
    )
