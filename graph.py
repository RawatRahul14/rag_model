# === Modules ===
from langgraph.graph import StateGraph, START, END

from rag_model.components.agent_state import AgentState
from rag_model.components.query_rewriter import query_rewriter_node
from rag_model.components.get_docs import doc_retriever
from rag_model.components.generate_answer import generate_answer_node

# === Build Graph ===
def build_graph():

    graph_builder = StateGraph(AgentState)

    ## === Add Nodes ===
    graph_builder.add_node("query_rewriter_node", query_rewriter_node)
    graph_builder.add_node("doc_retriever_node", doc_retriever)
    graph_builder.add_node("generate_answer_node", generate_answer_node)

    ## === Define Edges ===
    graph_builder.add_edge(START, "query_rewriter_node")
    graph_builder.add_edge("query_rewriter_node", "doc_retriever_node")
    graph_builder.add_edge("doc_retriever_node", "generate_answer_node")
    graph_builder.add_edge("generate_answer_node", END)

    return graph_builder.compile()

# === Compiled Graph ===
rag_graph = build_graph()