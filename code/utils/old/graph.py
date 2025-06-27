"""
graph.py – pre-build QA chains with configurable Pinecone index

Run two RetrievalQA nodes in parallel, timing each, and print total.

Usage:
    python graph.py [--case-id CASE_ID] [--index-name INDEX] [--buyer-model MODEL] [--stmt-model MODEL]

Examples:
    python graph.py --case-id 23CHLC22869 --index-name lasctesttwo
"""
import asyncio, time
import argparse
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model

from retriever import get_retriever

# ───────────────── Shared state ────────────────────────────────────────────
class S(TypedDict, total=False):
    case_id:    str
    topk_buyer: int
    topk_stmt:  int
    buyer:      str
    stmt:       str
    buyer_time: float
    stmt_time:  float
    buyer_src:  list[str]      # source page_file for buyer answer
    stmt_src:   list[str]      # source page_file for stmt answer

async def main(
    case_id: str,
    index_name: str,
    buyer_model: str,
    stmt_model: str,
    buyer_k: int,
    stmt_k: int,
):
    # 1) Pre-build chains synchronously
    buyer_chain = RetrievalQA.from_chain_type(
        llm=init_chat_model(buyer_model, temperature=0),
        retriever=get_retriever(
            index_name=index_name,
            case_id=case_id,
            doc_type="complaint",
            top_k=buyer_k
        ),
        return_source_documents=True,
    )
    stmt_chain = RetrievalQA.from_chain_type(
        llm=init_chat_model(stmt_model, temperature=0),
        retriever=get_retriever(
            index_name=index_name,
            case_id=case_id,
            doc_type=None,
            top_k=stmt_k
        ),
        return_source_documents=True,
    )

    # 2) Define pure-async nodes
    async def ask_buyer(state: S) -> S:
        t0 = time.perf_counter()
        out = await buyer_chain.ainvoke({"query": "Is the plaintiff a debt buyer?"})
        # extract page_file metadata from source_documents
        buyer_src = [doc.metadata.get("page_file") for doc in out.get("source_documents", []) if doc.metadata.get("page_file")]
        return {"buyer": out.get("result"), "buyer_time": time.perf_counter() - t0, "buyer_src": buyer_src}

    async def ask_stmt(state: S) -> S:
        t0 = time.perf_counter()
        out = await stmt_chain.ainvoke({"query": "Does the complaint contain a short statement?"})
        # extract page_file metadata from source_documents
        stmt_src = [doc.metadata.get("page_file") for doc in out.get("source_documents", []) if doc.metadata.get("page_file")]
        return {"stmt": out.get("result"), "stmt_time": time.perf_counter() - t0, "stmt_src": stmt_src}

    # 3) Build & compile graph
    g = StateGraph(S)
    g.add_node("buyer_node", ask_buyer)
    g.add_node("stmt_node", ask_stmt)
    g.add_edge(START, "buyer_node").add_edge(START, "stmt_node")
    g.add_edge("buyer_node", END).add_edge("stmt_node", END)
    graph = g.compile()

    # 4) Run & measure
    total_t0 = time.perf_counter()
    result = await graph.ainvoke({
        "case_id":    case_id,
        "topk_buyer": buyer_k,
        "topk_stmt":  stmt_k,
    })
    total_dur = time.perf_counter() - total_t0

    print(f"Debt buyer → {result['buyer']}   ({result['buyer_time']:.2f}s)")
    print(f"Short stmt → {result['stmt']}   ({result['stmt_time']:.2f}s)")
    # print source pages
    print("Sources for buyer answer:", result.get('buyer_src', []))
    print("Sources for stmt answer:", result.get('stmt_src', []))
    print(f"Total elapsed: {total_dur:.2f}s")

    # optionally generate HTML report and open in browser
    report_html = f"""
    <html>
      <head>
        <title>QA Graph Results for {case_id}</title>
      </head>
      <body>
        <h1>Case {case_id} Results</h1>
        <p><strong>Debt buyer:</strong> {result['buyer']} <em>({result['buyer_time']:.2f}s)</em></p>
        <p><strong>Sources for debt buyer:</strong> <span style=\"color:red\">{', '.join(result.get('buyer_src', []))}</span></p>
        <p><strong>Short statement:</strong> {result['stmt']} <em>({result['stmt_time']:.2f}s)</em></p>
        <p><strong>Sources for short statementt:</strong> <span style=\"color:red\">{', '.join(result.get('stmt_src', []))}</span></p>
        <p><strong>Total elapsed:</strong> {total_dur:.2f}s</p>
      </body>
    </html>
    """

    import webbrowser, tempfile, os
    fd, path = tempfile.mkstemp(suffix='.html', text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(report_html)
    webbrowser.open('file://' + path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id",      required=True, help="Case ID to query")
    parser.add_argument("--index-name",   default="lasc", help="Pinecone index name")
    parser.add_argument("--buyer-model",  default="openai:gpt-4.1", help="Model for buyer node")
    parser.add_argument("--stmt-model",   default="google_genai:gemini-1.5-flash-latest", help="Model for stmt node")
    parser.add_argument("--topk-buyer",   type=int, default=5, help="TopK for buyer retriever")
    parser.add_argument("--topk-stmt",    type=int, default=3, help="TopK for stmt retriever")
    args = parser.parse_args()

    asyncio.run(main(
        case_id=args.case_id,
        index_name=args.index_name,
        buyer_model=args.buyer_model,
        stmt_model=args.stmt_model,
        buyer_k=args.topk_buyer,
        stmt_k=args.topk_stmt,
    ))
