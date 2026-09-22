"""Pydantic schemas for validating structured LLM output."""

from pydantic import BaseModel, ConfigDict
import re

class AnswerResponse(BaseModel):
    """The exact structured response expected from a model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str
    confidence: float
    sources: list[str]


def parse_answer_response(response: str) -> AnswerResponse:
    """Parse and validate a raw JSON model response against AnswerResponse."""
    return AnswerResponse.model_validate_json(response)


def repair_json_response(response: str) -> str:
    """Apply narrowly scoped repairs to common JSON formatting defects."""
    start_index = response.find("{")
    end_index = response.rfind("}")
    if start_index == -1 or end_index == -1 or start_index > end_index:
        raise ValueError("No valid JSON object found")
    response = response[start_index:end_index+1]
    response = re.sub(r",\s*([\]}])", r"\1", response)
    return response
