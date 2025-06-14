# utils/retriever.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

def get_qa_chain(
    index_name: str,
    case_id: str,
    llm: ChatOpenAI = ChatOpenAI(model="gpt-4", temperature=0),
    embedding: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small"),
) -> RetrievalQA:
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embedding)

    # tell it to use our Document.page_content field
    retriever = vectorstore.as_retriever(
        text_key="page_content",
        search_kwargs={"filter": {"case_id": case_id}}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

