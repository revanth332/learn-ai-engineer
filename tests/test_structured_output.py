import pytest
from pydantic import ValidationError

from src.structured_output import AnswerResponse, parse_answer_response


def test_parses_a_valid_json_response():
    response = '{"answer": "BPE merges frequent pairs.", "confidence": 0.9, "sources": ["notes"]}'

    parsed = parse_answer_response(response)

    assert parsed == AnswerResponse(
        answer="BPE merges frequent pairs.",
        confidence=0.9,
        sources=["notes"],
    )


def test_rejects_invalid_json():
    with pytest.raises(ValidationError, match="json_invalid"):
        parse_answer_response('{"answer": }')


def test_rejects_non_object_json():
    with pytest.raises(ValidationError, match="model_type"):
        parse_answer_response('["not", "an", "object"]')


def test_rejects_missing_required_field():
    response = '{"answer": "BPE", "confidence": 0.9}'

    with pytest.raises(ValidationError, match="sources"):
        parse_answer_response(response)


def test_rejects_unexpected_field():
    response = '{"answer": "BPE", "confidence": 0.9, "sources": [], "extra": true}'

    with pytest.raises(ValidationError, match="extra"):
        parse_answer_response(response)


def test_rejects_coerced_string_for_strict_float_field():
    response = '{"answer": "BPE", "confidence": "0.9", "sources": []}'

    with pytest.raises(ValidationError, match="confidence"):
        parse_answer_response(response)


def test_rejects_non_string_source_items():
    response = '{"answer": "BPE", "confidence": 0.9, "sources": [123]}'

    with pytest.raises(ValidationError, match="sources"):
        parse_answer_response(response)


def test_generates_a_json_schema_for_provider_schema_enforcement():
    schema = AnswerResponse.model_json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"answer", "confidence", "sources"}
