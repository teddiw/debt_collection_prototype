"""
cited_output_graph.py – pre-build QA chains with configurable Pinecone index

Run all RetrievalQA nodes and cited output generation in parallel.

Usage:
    python cited_output_graph.py [--case-id CASE_ID] [--index-name INDEX] [--buyer-model MODEL] [--stmt-model MODEL]

Examples:
    python cited_output_graph.py --case-id 23CHLC22869 --index-name lasctesttwo
"""
import asyncio, time
import argparse
import operator
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END
from langchain.chains import RetrievalQA
from typing import Annotated

from citation_generation.cited_generation import cited_generation
from citation_generation.citation_utils import remove_citations
from graph_system.cited_output_nodes import construct_llms, construct_retrievers, single_node_chain_names, triple_node_chain_names, all_chain_names, all_node_names, node_to_top_k_mapping, needs_allegation_from_predecessor_node_names
from graph_system.query_mappings import node_to_retrieval_query_mapping, node_to_cited_response_query_mapping
import pandas as pd 

# ───────────────── Shared state ────────────────────────────────────────────
class S(TypedDict, total=False):
    case_id:    str
    top_k:      int
    retrieved_responses: Annotated[list, operator.add] # retrieved responses by node
    cited_responses:  Annotated[list, operator.add] # list[str]    # cited responses by node
    retrieved_source_pages:  Annotated[list, operator.add]# list[list[str]]  # retrieved page names by node
    cited_source_pages:  Annotated[list, operator.add] # list[list[str]]      # cited page names by node 
    cited_quotes:  Annotated[list, operator.add] # list[list[str]]     # cited quotes by node
    highlighted_sources:  Annotated[list, operator.add] # list[list[str]]  # sources with highlighted cited quotes by node
    highlighted_source_snippets:  Annotated[list, operator.add] # list[list[str]]  # highlighted cited quotes in source snippet by node
    requirement_satisfied:  Annotated[list, operator.add] # list[bool]  # whether the requirement is satisfied for the entire graph
    plaintiff: str  # name of the plaintiff
    defendant: str  # name of the defendant
    short_responses: Annotated[list, operator.add]  # short responses by node, used for chaining retrieval prompts that require the output of their predecessor


async def main(
    case_id: str,
    index_name: str,
    model_str: str="openai:gpt-4",
    top_k: int=3,
):
    
    # 1. Build LLMs and retrievers synchronously
    node_to_llm_mapping = construct_llms(model_str=model_str, temperature=0)
    node_to_retriever_mapping = construct_retrievers(node_to_llm_mapping, index_name, case_id, node_to_top_k_mapping)

    # 2) Define a function that generates pure-async nodes from node_name and mappings
    def build_node(node_name: str,
                   node_to_llm_mapping: dict[str, Optional[any]],
                   node_to_retriever_mapping: dict[str, Optional[RetrievalQA]],
                   node_to_retrieval_query_mapping: dict[str, str], 
                   node_to_cited_response_query_mapping: dict[str, str], 
                   ):
        
        async def node(state: S) -> S: 
            retrieval_query = node_to_retrieval_query_mapping[node_name]
            cited_response_query = node_to_cited_response_query_mapping[node_name].format(plaintiff=state["plaintiff"], defendant=state["defendant"])
            node_number = int(node_name.split("_")[-1])
            short_response = "" 
            if node_name in needs_allegation_from_predecessor_node_names:
                # This is a node where the retrieval and generation queries require the output of the predecessor node
                predecessor_node_name = node_name.split("_")[0] + "_"+str(node_number-1)
                for nn, cited_response in state['cited_responses']:
                    if nn == predecessor_node_name:
                        predecessor_response = cited_response
                        break
                predecessor_response = remove_citations(predecessor_response)
                cited_response_query += "\nInformation alleged by the plaintiff: " + predecessor_response
                for nn, curr_short_response in state['short_responses']:
                    if nn == predecessor_node_name:
                        short_response = curr_short_response
                        break
            retrieval_query = retrieval_query.format(plaintiff=state["plaintiff"], defendant=state["defendant"], short_allegation=short_response)
            print()
            print("RQ:", retrieval_query)
            print("GQ:", cited_response_query)
            print()
            out = await node_to_retriever_mapping[node_name].ainvoke({"query": retrieval_query})
            # Extract page names from source_documents
            source_pages = [doc.metadata.get("doc_type")+'/'+doc.metadata.get("page_file") for doc in out.get("source_documents", []) if doc.metadata.get("page_file")]
            # Get cited answer
            answer = cited_generation(cited_response_query, out['source_documents'], node_to_llm_mapping[node_name], verbose=True)

            return {'retrieved_responses': [(node_name, out.get("result"))],
                    'cited_responses': [(node_name, answer.cited_response)],
                    'retrieved_source_pages': [(node_name, source_pages)],
                    'highlighted_sources': [(node_name, answer.tag_highlighted_cited_sources)],
                    'highlighted_source_snippets': [(node_name, answer.tag_highlighted_cited_source_snippets)],
                    'cited_quotes': [(node_name, answer.cited_quotes)],
                    'cited_source_pages': [(node_name, answer.cited_source_names)],
                    'requirement_satisfied': [(node_name, answer.requirement_satisfied)],
                    'short_responses': [(node_name, answer.short_answer)],
                    }
        return node

    # 3) Build & compile graph
    g = StateGraph(S)
    for node_name in node_to_llm_mapping.keys():
        node = build_node(node_name,
                          node_to_llm_mapping,
                          node_to_retriever_mapping,
                          node_to_retrieval_query_mapping,
                          node_to_cited_response_query_mapping
                          )
        g.add_node(node_name, node)

    # 3a) Add all of the single-node chains
    for node_name in single_node_chain_names:
        g.add_edge(START, node_name+"_0").add_edge(node_name+"_0", END)

    # 3b) Add all of the triple-node chains 
    for node_name in triple_node_chain_names:
        g.add_edge(START, node_name+"_0").add_edge(node_name+"_0", node_name+"_1")
        g.add_edge(node_name+"_1", node_name+"_2").add_edge(node_name+"_2", END)

    # 3c) Add all of the two-node chains (right now, these are all chains besides single-node and triple-node)
    for node_name in all_chain_names:
        if (node_name not in single_node_chain_names) and (node_name not in triple_node_chain_names):
            g.add_edge(START, node_name+"_0").add_edge(node_name+"_0", node_name+"_1")
            g.add_edge(node_name+"_1", END)

    graph = g.compile()

    # Visualize the graph (paste printed output into https://mermaid.live)
    # print(graph.get_graph().draw_mermaid())

    # 4) Run & measure
    total_t0 = time.perf_counter()

    # 4.1) Extract basic facts about the case: identify the plaintiff and defendant and store in the state
    some_llm = node_to_llm_mapping["hasSignedSworn1788.60_0"]
    some_retriever = node_to_retriever_mapping["hasSignedSworn1788.60_0"]
    plaintiff_query = "What is the name of the plaintiff?"
    plaintiff_out = await some_retriever.ainvoke({"query": plaintiff_query})
    plaintiff = cited_generation(plaintiff_query, plaintiff_out['source_documents'], some_llm, verbose=False).short_answer
    defendant_query = "What is the name of the defendant?"
    defendant_out = await some_retriever.ainvoke({"query": defendant_query})
    defendant = cited_generation(defendant_query, defendant_out['source_documents'], some_llm, verbose=False).short_answer

    state = {
        "case_id": case_id,
        "top_k": top_k,
        "retrieved_responses": [],   
        "cited_responses": [],   
        "retrieved_source_pages": [],  
        "cited_source_pages":  [],
        "cited_quotes": [],     
        "highlighted_sources": [],  
        "highlighted_source_snippets": [],
        "requirement_satisfied": [],
        "plaintiff": plaintiff,
        "defendant": defendant,  
    }
    result = await graph.ainvoke(state)
    total_dur = time.perf_counter() - total_t0

    print(f"Total elapsed: {total_dur:.2f}s")

    # 5) Save results to a one-row pandas table in csv format
    # 5.1) Flatten the results into a dictionary
    result_dict = {}
    for key_i, value_i in result.items():
        if not isinstance(value_i, list):
            result_dict[key_i] = [value_i]
        else:
            for key_j, value_j in value_i:
                result_dict[f"{key_i}:{key_j}"] = [value_j]
    
    df = pd.DataFrame(result_dict)
    df.to_csv(f"results/{case_id}_system_output.csv", index=False)

    # 6) Optionally generate HTML report and open in browser
    report_html = f"""
    <html>
      <head>
        <title>QA Graph Results for {case_id}</title>
      </head>
      <body>
        <h1>Case {case_id} Results</h1>
      <p><strong>Total time elapsed:</strong> {total_dur:.2f}s</p>"""
    
    for node_name in all_node_names:
        
        requirement_satisfied = result_dict[f"requirement_satisfied:{node_name}"][0]
        if requirement_satisfied:
            requirement_satisfied_str = "<p><strong><span style=\"color: #107ce8\">Requirement Satisfied</span></strong></p>"
        else:
            requirement_satisfied_str = "<p><strong><span style=\"color: #e810e1\">Requirement Not Satisfied</span></strong></p>"

        cited_sources = result_dict[f"cited_source_pages:{node_name}"][0]
        
        cited_quote = result_dict[f"cited_quotes:{node_name}"][0]
        for i in range(len(cited_quote)):
            cited_quote[i] = f"[{i+1}] "+cited_quote[i]
        cited_quotes_str = "<br>"+"<br>".join(cited_quote)
        cited_quotes_str = f"<span style=\"color: #13872a\">{cited_quotes_str}</span>"

        cited_quote_snippets = result_dict[f"highlighted_source_snippets:{node_name}"][0]
        for i in range(len(cited_quote_snippets)):
            cited_quote_snippets[i] = cited_quote_snippets[i].replace("<highlight>", "<span style=\"color: #13872a\">").replace("</highlight>", "</span>")
            cited_quote_snippets[i] = f"<span style=\"color: #13872a\">[{i+1}]</span> "+cited_quote_snippets[i]
        cited_quotes_snippet_str = "<br>"+"<br>".join(cited_quote_snippets) # can use below for "Cited quotes"

        report_html += f"""
        <p><strong>{node_to_cited_response_query_mapping[node_name].format(plaintiff=result["plaintiff"], defendant=result["defendant"])}</strong></p> 
        {requirement_satisfied_str}
        <p><strong>Cited response: </strong>{result_dict[f"cited_responses:{node_name}"][0].replace('\n', '<br>')}</p>
        <p><strong>Cited quotes:</strong>{cited_quotes_str}</p> 
        <p><strong>Cited sources:</strong> {", ".join(cited_sources)}</p>
        <p><strong>Uncited response: </strong>{result_dict[f"retrieved_responses:{node_name}"][0].replace('\n', '<br>')}</p>
        <p><strong>Sources: </strong>{", ".join(result_dict[f"retrieved_source_pages:{node_name}"][0])}</p>
        <hr>
    """
    report_html += """
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
    parser.add_argument("--case_id",      required=True, help="Case ID to query")
    parser.add_argument("--index_name",   default="lasctesttwo", help="Pinecone index name")
    parser.add_argument("--model_str",  default="openai:gpt-4.1", help="Model for buyer node")
    parser.add_argument("--top_k",   type=int, default=4, help="Top K for the retriever for all nodes")
    args = parser.parse_args()

    asyncio.run(main(
        case_id=args.case_id,
        index_name=args.index_name,
        model_str=args.model_str,
        top_k=args.top_k,
    ))
