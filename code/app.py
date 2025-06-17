#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

from code.utils.pdf_to_markdown_pipeline import batch_process_local_pdfs
from utils.embed_and_store       import embed_ocr_pages_to_pinecone
from utils.retriever             import get_qa_chain
from utils.query                 import send_query

def main(case_id: str, index_name: str, question: str):
    load_dotenv()  # load OPENAI_API_KEY, PINECONE_API_KEY, etc.

    case_dir = os.path.join("files", "cases_parsed", case_id)
    if not os.path.isdir(case_dir):
        print(f"[ERROR] Case folder not found: {case_dir}")
        sys.exit(1)

    # (Skip OCR if already done)
    # batch_process_local_pdfs("files/raw_cases")

    # Step 2: embed all pages of this case (all doc types) into Pinecone
    embed_ocr_pages_to_pinecone(
        case_dir=case_dir,
        pinecone_index_name=index_name,
    )

    # Step 3: build a RetrievalQA chain that filters to this case_id
    qa_chain = get_qa_chain(
        index_name=index_name,
        case_id=case_id
    )

    # Step 4: ask your question
    send_query(qa_chain, question)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python code/app.py <CASE_ID> <Your question here>")
        sys.exit(1)

    case_id  = sys.argv[1]
    question = " ".join(sys.argv[2:])
    INDEX    = "lasc"  # or your Pinecone index name

    main(case_id, INDEX, question)
