import pytest

from src.structured_output import parse_answer_response, repair_json_response


VALID_RESPONSE = '{"answer": "BPE", "confidence": 0.9, "sources": ["notes"]}'


def test_returns_already_valid_json_unchanged():
    assert repair_json_response(VALID_RESPONSE) == VALID_RESPONSE


def test_removes_a_json_markdown_code_fence():
    response = "```json\n" + VALID_RESPONSE + "\n```"

    assert repair_json_response(response) == VALID_RESPONSE


def test_extracts_json_object_surrounded_by_explanatory_text():
    response = "Here is the requested result:\n" + VALID_RESPONSE + "\nI hope this helps."

    assert repair_json_response(response) == VALID_RESPONSE


def test_removes_a_trailing_comma_from_an_object():
    response = '{"answer": "BPE", "confidence": 0.9, "sources": ["notes"],}'

    repaired = repair_json_response(response)

    assert parse_answer_response(repaired).answer == "BPE"


def test_removes_a_trailing_comma_from_an_array():
    response = '{"answer": "BPE", "confidence": 0.9, "sources": ["notes",]}'

    repaired = repair_json_response(response)

    assert parse_answer_response(repaired).sources == ["notes"]


def test_removes_trailing_comma_with_whitespace_before_closing_bracket():
    response = '{"answer": "BPE", "confidence": 0.9, "sources": ["notes", \n ]}'

    repaired = repair_json_response(response)

    assert parse_answer_response(repaired).sources == ["notes"]


def test_preserves_commas_and_braces_inside_json_strings():
    response = (
        'Explanation: {"answer": "Use {braces}, commas, and [brackets].", '
        '"confidence": 0.9, "sources": ["notes"]} Thanks!'
    )

    repaired = repair_json_response(response)

    assert parse_answer_response(repaired).answer == "Use {braces}, commas, and [brackets]."


def test_rejects_text_without_a_json_object():
    with pytest.raises(ValueError, match="JSON object"):
        repair_json_response("The model did not return JSON.")


def test_does_not_repair_schema_errors():
    response = '{"answer": "BPE", "confidence": 0.9}'

    repaired = repair_json_response(response)

    with pytest.raises(Exception):
        parse_answer_response(repaired)
