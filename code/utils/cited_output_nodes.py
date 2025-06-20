from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from retriever import get_retriever

# Keep a list of the node names
# Node names programmatically correspond to the nodes they're connected to
all_chain_names = [
    "hasSignedSworn1788.60",
    "hasShortStatement",
    "isDebtBuyer", # two nodes _0 and _1
    "isSoleOwner", # two nodes _0 and _1
    "hasChargeOffBalance", # two nodes _0 and _1
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

complaint_chain_names = [
    # The answer to each of these nodes is based on the complaint
    "hasShortStatement_0",
    "isDebtBuyer_0", 
    "isSoleOwner_0", 
    "hasChargeOffBalance_0", 
    "defaultOrLastPaymentDate_0", 
    "defaultOrLastPaymentDate_2",
    "chargeOffCreditorInfo_0", 
    "debtorInfo_0", 
    "postChargeOffPurchaserInfo_0", 
    "is1788.52Compliant_0",
    "hasContractOrLastStatement_0"
]

other_doc_chain_names = [
    # The answer to each of these nodes is based on the complaint
    "hasSignedSworn1788.60_0",
    "isDebtBuyer_1", 
    "isSoleOwner_1", 
    "hasChargeOffBalance_1", 
    "defaultOrLastPaymentDate_1",  
    "chargeOffCreditorInfo_1", 
    "debtorInfo_1", 
    "postChargeOffPurchaserInfo_1", 
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

def construct_retrievers(node_to_llm_mapping, index_name, case_id, top_k=5):
    """ Initialize the retrievers for each node in the graph. Sequential nodes share the same retriever instance.
    TODO Enable different nodes to use different top_k values
    """
    node_to_retriever_mapping = {}
    node_names = list(node_to_llm_mapping.keys())
    for node_name in node_names:
        if node_name in complaint_chain_names:
            doc_type = "complaint"
        else:
            doc_type = "other"
        retriever = RetrievalQA.from_chain_type(
                                                llm=node_to_llm_mapping[node_name],
                                                retriever=get_retriever(
                                                    index_name=index_name,
                                                    case_id=case_id,
                                                    doc_type=doc_type,
                                                    top_k=top_k
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




