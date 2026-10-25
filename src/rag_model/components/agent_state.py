# === Modules ===
from typing import TypedDict

class AgentState(TypedDict):

    question: str
    previous_question: str
    rewritten_question: str

    documents: list | None

    answer: str