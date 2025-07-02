import asyncio

from code.utils.cited_output_graph import main
from code.utils.document_storage.embed_and_store import embed_and_store

if __name__ == "__main__":
    model_str = "openai:gpt-4"
    top_k = 3
    index_name = "lasctesttwo"
    case_ids = [
        "23CHLC12118",
        "23CHLC18998",
        "23CHLC22869", # R1 Triaged 
        "23CHLC26147",
        "24NWLC34626",
        "24NWLC38337"
    ]

    for case_id in case_ids:
        embed_and_store(
            case_dir=f"files/cases_parsed/{case_id}",
            pinecone_index_name="lasctesttwo",
        )

        asyncio.run(main(
            case_id=case_id,
            index_name=index_name,
            model_str=model_str,
            top_k=top_k,
        ))

# ["23CHLC12118", "23CHLC18998", "23CHLC22869", "23CHLC26147", "24NWLC34626", "24NWLC38337"]
# python code/utils/embed_and_store.py --case-dir files/cases_parsed/24NWLC38337 --index-name lasctesttwo
