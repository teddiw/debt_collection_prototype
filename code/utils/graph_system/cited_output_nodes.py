from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from .retriever import get_retriever

# Keep a list of the node names
# Node names programmatically correspond to the nodes they're connected to
all_chain_names = [
    "hasSignedSworn1788.60",
    "hasShortStatement",
    "isDebtBuyer", # two nodes _0 and _1
    "isSoleOwner", # two nodes _0 and _1
    "hasChargeOffBalance", # two nodes _0 and _1
    "hasPostChargeOffFeeExplanation", # two nodes _0 and _1
    "defaultOrLastPaymentDate", # three nodes _0, _1, and _2
    "chargeOffCreditorInfo", # two nodes _0 and _1
    "debtorInfo", # two nodes _0 and _1
    "postChargeOffPurchaserInfo", # two nodes _0 and _1
    "is1788.52Compliant",
    "hasContractOrLastStatement",
]

single_node_chain_names = [
    "hasSignedSworn1788.60",
    "hasShortStatement",
    "is1788.52Compliant",
    "hasContractOrLastStatement",
]

triple_node_chain_names = [
    "defaultOrLastPaymentDate",
]

# Assume all other chains besides those in single_node_chain_names and triple_node_chain_names have two nodes, suffixed with _0 and _1

complaint_allegation_node_names = [
    # The answer to each of these nodes is based on the complaint
    "hasShortStatement_0",
    "isDebtBuyer_0", 
    "isSoleOwner_0", 
    "hasChargeOffBalance_0", 
    "hasPostChargeOffFeeExplanation_0",
    "defaultOrLastPaymentDate_0", 
    "defaultOrLastPaymentDate_2",
    "chargeOffCreditorInfo_0", 
    "debtorInfo_0", 
    "postChargeOffPurchaserInfo_0", 
    "is1788.52Compliant_0"
]

other_doc_node_names = [
    # The answer to each of these nodes is based on documents classified as "other" (e.g., declarations)
    "hasSignedSworn1788.60_0",
]

evidence_node_names = [
    # The answer to each of these nodes is currently based on all documents
    "isDebtBuyer_1", 
    "isSoleOwner_1", 
    "hasChargeOffBalance_1", 
    "hasPostChargeOffFeeExplanation_1",
    "defaultOrLastPaymentDate_1",  
    "chargeOffCreditorInfo_1", 
    "debtorInfo_1", 
    "postChargeOffPurchaserInfo_1", 
    "hasContractOrLastStatement_0",
]

def construct_llms(model_str, temperature=0):
    """ Initialize the LLMs for each node in the graph. Sequential nodes share the same LLM instance.
    TODO Enable different nodes to use different LLMs
    """
    node_to_llm_mapping = {}
    for chain_name in all_chain_names:
        llm = init_chat_model(model_str, temperature=temperature)
        if chain_name in single_node_chain_names:
            # chain name + "_0" is the node name
            node_to_llm_mapping[chain_name+"_0"] = llm
        elif chain_name in triple_node_chain_names:
            # chain name + "_0", chain name + "_1", and chain name + "_2" are the node names
            node_to_llm_mapping[chain_name + "_0"] = llm
            node_to_llm_mapping[chain_name + "_1"] = llm
            node_to_llm_mapping[chain_name + "_2"] = llm
        else:
            # chain name + "_0" and chain name + "_1" are the node names
            node_to_llm_mapping[chain_name + "_0"] = llm
            node_to_llm_mapping[chain_name + "_1"] = llm
    return node_to_llm_mapping

def construct_retrievers(node_to_llm_mapping, index_name, case_id, node_to_top_k_mapping):
    """ Initialize the retrievers for each node in the graph. Sequential nodes share the same retriever instance.
    """
    node_to_retriever_mapping = {}
    node_names = list(node_to_llm_mapping.keys())
    for node_name in node_names:
        if node_name in complaint_allegation_node_names:
            doc_type = "complaint"
            exhibit_or_allegation = None
        elif node_name in other_doc_node_names:
            doc_type = "other"
            exhibit_or_allegation = None
        elif node_name in evidence_node_names:
            doc_type = None
            exhibit_or_allegation = "exhibit"
        else:
            print(f"[WARNING] Node {node_name} does not have a specific doc_type or exhibit_or_allegation defined. Defaulting to None.")
            doc_type = None
            exhibit_or_allegation = None
        retriever = RetrievalQA.from_chain_type(
                                                llm=node_to_llm_mapping[node_name],
                                                retriever=get_retriever(
                                                    index_name=index_name,
                                                    case_id=case_id,
                                                    doc_type=doc_type,
                                                    exhibit_or_allegation=exhibit_or_allegation,
                                                    top_k=node_to_top_k_mapping[node_name]
                                                ),
                                                return_source_documents=True,
                                            )
        node_to_retriever_mapping[node_name] = retriever

    return node_to_retriever_mapping
    
all_node_names = [
    "hasSignedSworn1788.60_0",
    "hasShortStatement_0",
    "isDebtBuyer_0", 
    "isDebtBuyer_1",
    "isSoleOwner_0", 
    "isSoleOwner_1",
    "hasChargeOffBalance_0", 
    "hasChargeOffBalance_1",
    "hasPostChargeOffFeeExplanation_0",
    "hasPostChargeOffFeeExplanation_1",
    "defaultOrLastPaymentDate_0", 
    "defaultOrLastPaymentDate_1",
    "defaultOrLastPaymentDate_2",
    "chargeOffCreditorInfo_0", 
    "chargeOffCreditorInfo_1",
    "debtorInfo_0", 
    "debtorInfo_1",
    "postChargeOffPurchaserInfo_0", 
    "postChargeOffPurchaserInfo_1",
    "is1788.52Compliant_0",
    "hasContractOrLastStatement_0",
]

node_to_top_k_mapping = {
    "hasSignedSworn1788.60_0": 4,
    "hasShortStatement_0": 4,
    "isDebtBuyer_0": 4,
    "isDebtBuyer_1": 4,
    "isSoleOwner_0": 4,
    "isSoleOwner_1": 4,
    "hasChargeOffBalance_0": 4,
    "hasChargeOffBalance_1": 20,
    "hasPostChargeOffFeeExplanation_0": 4,
    "hasPostChargeOffFeeExplanation_1": 20,
    "defaultOrLastPaymentDate_0": 4,
    "defaultOrLastPaymentDate_1": 20,
    "defaultOrLastPaymentDate_2": 4,
    "chargeOffCreditorInfo_0": 4,
    "chargeOffCreditorInfo_1": 4,
    "debtorInfo_0": 4,
    "debtorInfo_1": 4,
    "postChargeOffPurchaserInfo_0": 4,
    "postChargeOffPurchaserInfo_1": 10,
    "is1788.52Compliant_0": 4,
    "hasContractOrLastStatement_0": 10,
}

needs_allegation_from_predecessor_node_names = [
    "hasChargeOffBalance_1",
    "hasPostChargeOffFeeExplanation_1",
    "defaultOrLastPaymentDate_1",
    "defaultOrLastPaymentDate_2",
    "chargeOffCreditorInfo_1",
    "debtorInfo_1",
    "postChargeOffPurchaserInfo_1",
]




