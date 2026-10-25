# === Modules ===
from langchain_openai import ChatOpenAI

from rag_model.components.agent_state import AgentState
from rag_model.components.models import Rewriter


# === Query Rewriter Node ===
def query_rewriter_node(state: AgentState) -> AgentState:
    """
    Rewrites the current question into a standalone retrieval query,
    using the previous question when necessary.
    """

    ## === Extract Questions ===
    current_question = state.get("question", "")
    previous_question = state.get("previous_question", "")

    ## === Initialize LLM ===
    llm_model = ChatOpenAI(
        model = "gpt-4o-mini",
        temperature = 0,
    ).with_structured_output(Rewriter)

    ## === Prompt ===
    prompt = f"""
    You are a query-rewriting assistant for a document retrieval system.

    Rewrite the current question into a clear, standalone query suitable
    for retrieving relevant information from a document.

    Rules:
    - If the current question is self-contained, preserve its meaning.
    - If it depends on the previous question, resolve references using the previous question.
    - Do not answer the question.
    - Do not introduce facts that are not present in the questions.

    Current question:
    {current_question}

    Previous question:
    {previous_question or "None"}
    """

    ## === Rewrite Query ===
    response = llm_model.invoke(prompt)

    ## === Update State ===
    return {
        **state,
        "previous_question": current_question,
        "documents": [],
        "rewritten_question": response.rephrased_query,
    }