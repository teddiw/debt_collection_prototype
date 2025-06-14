# utils/query.py

def send_query(
    qa_chain,
    question: str,
    verbose: bool = True
):
    """
    Runs `question` through the RetrievalQA chain.
    Returns (answer, source_docs).
    """
    # use invoke() to avoid the deprecation warning
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]

    if verbose:
        print("\n── ANSWER ─────────────────────────────────────────")
        print(answer)
        print("\n── SOURCE DOCUMENTS ─────────────────────────────")
        for doc in sources:
            meta = doc.metadata
            src = meta.get("source", meta.get("source_page", "<unknown>"))
            print(f"{meta.get('case_id','?')}/{meta.get('doc_type','?')}/{src}")

    return answer, sources
