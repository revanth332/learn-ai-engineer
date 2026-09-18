"""Pydantic schemas for validating structured LLM output."""

from pydantic import BaseModel, ConfigDict


class AnswerResponse(BaseModel):
    """The exact structured response expected from a model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str
    confidence: float
    sources: list[str]


def parse_answer_response(response: str) -> AnswerResponse:
    """Parse and validate a raw JSON model response against AnswerResponse."""
    return AnswerResponse.model_validate_json(response)
