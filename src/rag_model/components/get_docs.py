# === Modules ===
from langchain_core.documents import Document

from rag_model.components.agent_state import AgentState

# === Document Retriever Node ===
def doc_retriever(
    state: AgentState,
    *,
    config,
) -> AgentState:

    # === Get Retriever ===
    retriever = config.get("configurable", {}).get("retriever")

    if retriever is None:
        raise ValueError("Retriever not provided in graph configuration.")

    # === Get Query ===
    query = state.get("rewritten_question") or state.get("question", "")

    if not query:
        raise ValueError("No query provided for retrieval.")

    # === Retrieve Documents ===
    documents = retriever.invoke(query)

    # === Update State ===
    return {
        **state,
        "documents": documents,
    }