# === Modules ===
from langchain_core.runnables import RunnableConfig

from rag_model.components.agent_state import AgentState

# === Document Retriever ===
def doc_retriever(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> AgentState:

    ## === Get the retriever passed through graph configuration. ===
    retriever = config.get("configurable", {}).get("retriever")

    if retriever is None:
        raise ValueError("Retriever not provided in graph configuration.")

    ## === Prefer the rewritten query for document retrieval. ===
    query = state.get("rewritten_question") or state.get("question", "")

    if not query:
        raise ValueError("No query provided for retrieval.")

    documents = retriever.invoke(query)

    return {
        **state,
        "documents": documents,
    }