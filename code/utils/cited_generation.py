# utils/cited_generation.py

import copy

from langchain_core.prompts       import ChatPromptTemplate

from generation_classes     import QuoteExtraction, CitedOutputGeneration
from citation_utils         import find_quote, highlight_source
from answer                 import Answer

def cited_generation(question, doc_ls, llm, verbose=True):
    """
    Generates a cited response to the question using the provided source documents.
    
    Arguments:
    - question: The question to be answered.
    - doc_ls: A list of source documents retrieved from the vector DB for the query.
    - llm: The language model for generating the cited response.

    Returns:
    Answer object (see answer.py)
    """
    # Extract the relevent spans of the source
    source_ls = [src.page_content.replace(r'\.', '.') for src in doc_ls]  # Extract the text content from the source documents
    source_ls_str = "\n\n".join([f"Source {i+1}:\n{src}" for i, src in enumerate(source_ls)])
    quote_extraction_llm = llm.with_structured_output(QuoteExtraction)
    few_shot_prompt = ChatPromptTemplate.from_messages([("system", QuoteExtraction.system_prompt), ("human", "{input}")])
    quote_extraction_llm = few_shot_prompt | quote_extraction_llm

    source_quote_ls = quote_extraction_llm.invoke(f"Query:{question}\nSources:\n{source_ls_str}").source_quote_ls
    source_quote_ls = [quote.strip() for quote in source_quote_ls if quote.strip()]  # Removes extra whitespace and any (unlikely) empty quotes

    # Using string-matching, verify that the quote is found in the original source. 
    verified_source_quote_ls = []  # List to store verified quotes
    for quote in source_quote_ls:
        found_quote = False
        for i in range(len(source_ls)):
            source = source_ls[i]        
            if find_quote(quote, source):
                found_quote = True
                verified_source_quote_ls.append(quote)
                break
        if (found_quote == False): # TODO examine these failure modes; is find_quote too conservative? Enter a retry loop (raise temp)? 
            print(f"Error: Quote '{quote}' not found in any of the original sources. It was not verified and will not be used in generation.")
    # Check whether there are verified quotes to used for generation
    if (len(verified_source_quote_ls) == 0):
        requirement_satisfied = False # The information to answer the query is missing. The verifiability tool for this is under consideration.
    else:
        requirement_satisfied = True # The verifiability tool for this is citation to the extracted quote in the source.

    # Generate the final cited response using the verified quotes
    numbered_source_quote_ls = "\n".join([f"Quote #{i+1}: {quote}" for i, quote in enumerate(verified_source_quote_ls)]) # number the verified quotes

    cited_output_generation_llm = llm.with_structured_output(CitedOutputGeneration)
    few_shot_prompt = ChatPromptTemplate.from_messages([("system", CitedOutputGeneration.system_prompt), ("human", "{input}")])
    cited_output_generation_llm = few_shot_prompt | cited_output_generation_llm

    answer_dict = {} # TODO Custom to each question, save the relevant information for Gate 2 in answer_dict (i.e., the debtor's name and address). Update requirement_satisfied.

    cited_output_text = cited_output_generation_llm.invoke(
        f"Query: {question}\nSource quotes:\n{numbered_source_quote_ls}" # TODO use the highlighted sources when generating for more context
    ).cited_output_text 

    if (cited_output_text == "No answer."):
        requirement_satisfied = False

    # Create the Answer object
    answer = Answer(question,
                    cited_output_text,
                    verified_source_quote_ls,
                    source_ls,
                    requirement_satisfied,
                    answer_dict
                    )
    
    if verbose:
        source_quotes_str = "\n".join([f"Quote #{i+1}: {quote}" for i, quote in enumerate(answer.cited_quotes)])
        print("\n── CITED ANSWER ─────────────────────────────────────────")
        print(f"{answer.cited_response}\n")
        print("\n── CITED SOURCE QUOTES ─────────────────────────────────────────")
        print(f"{source_quotes_str}\n")
        # TODO implement answer.cited_source_names in answer.py
        # print("\n── CITED SOURCE NAMES ─────────────────────────────────────────")
        # print(f"{'\n'.join(answer.cited_source_names)}")
        print("\n── HIGHLIGHTED CITED SOURCES ─────────────────────────────────────────")
        print(f"{'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'.join(answer.color_highlighted_cited_sources)}")

    return answer



        



    
