# === Modules ===
from pydantic import BaseModel, Field

# === Rewriter ===
class Rewriter(BaseModel):
    rephrased_query: str = Field(
        description = "Rephrased query made by the model"
    )