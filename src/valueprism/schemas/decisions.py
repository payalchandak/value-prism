# Copied from values-schemas/values_schemas/decisions.py.
# The import below is the only change: it points at this package.
"""Models for LLM decision evaluation."""

from pydantic import BaseModel, Field
from typing import Literal, Optional

from valueprism.schemas.case import BenchmarkCandidate


class RunResult(BaseModel):
    """A single run result from evaluating an LLM on a case.
    
    Stores the complete LLM response dict from all-the-llms, which includes:
    - Response text (choices[0].message.content)
    - Model version (model field)
    - Token usage and cost (usage field)
    - Timestamp (created field)
    - And any other metadata the LLM provider returns
    """
    
    full_response: dict = Field(
        ..., 
        description="Full response dict from all-the-llms completion (includes model, usage, etc.)"
    )
    parsed_choice: Literal["choice_1", "choice_2", "REFUSAL", "INCOHERENT"] = Field(
        ...,
        description=(
            "Extracted choice from the response. 'REFUSAL' is a coherent, "
            "intentional non-selection; 'INCOHERENT' is a response that "
            "cannot be meaningfully interpreted or mapped to a choice "
            "(e.g. generation breakdown under heavy steering)."
        )
    )


class ModelDecisionData(BaseModel):
    """Data for a single model's evaluation on a case."""
    
    temperature: float = Field(..., description="Temperature used for generation")
    runs: list[RunResult] = Field(default_factory=list, description="All run results")


class DecisionRecord(BaseModel):
    """Complete record of LLM decisions for a single case.
    
    One JSON file per case at data/llm_decisions/{case_id}.json
    
    This record is fully self-contained, embedding the complete case definition
    along with all model evaluations. This ensures that each record is an
    immutable snapshot - if the original case changes, the decision record
    preserves the exact vignette and choices that were presented to the models.
    """
    case_id: str = Field(..., description="Unique identifier for the case")
    case: BenchmarkCandidate = Field(
        ...,
        description="Complete case definition (vignette, choices, value tags)"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="The rendered system prompt used for LLM evaluation"
    )
    user_prompt: Optional[str] = Field(
        default=None,
        description="The rendered user prompt used for LLM evaluation"
    )
    models: dict[str, ModelDecisionData] = Field(
        default_factory=dict,
        description="Evaluation results keyed by model name (e.g., 'openai/gpt-4o')"
    )
