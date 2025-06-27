# retriever.py
from __future__ import annotations
import os
from typing import Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

__all__ = ["get_retriever"]
load_dotenv()

def get_retriever(
    *,
    index_name: str,
    case_id: str,
    doc_type: Optional[str] = None,
    embedding: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small"),
    namespace: Optional[str] = None,
    top_k: int = 6,
):
    """Return a Pinecone retriever filtered by case_id (and opt. doc_type)."""
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    store = PineconeVectorStore(
        index     = pc.Index(index_name),
        embedding = embedding,
        namespace = namespace,
    )
    filt = {"case_id": case_id}
    if doc_type:
        filt["doc_type"] = doc_type

    return store.as_retriever(
        search_kwargs={"filter": filt, "k": top_k}
    )
