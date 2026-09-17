import pytest

from src.tokenization.context import (
    ChatMessage,
    estimate_message_tokens,
    estimate_prompt_tokens,
    trim_messages,
)


def test_message_estimate_includes_utf8_bytes_and_framing():
    message = ChatMessage(role="user", content="hi")

    assert estimate_message_tokens(message) == 10


def test_prompt_estimate_includes_assistant_reply_primer():
    messages = [
        ChatMessage(role="system", content="be helpful"),
        ChatMessage(role="user", content="hi"),
    ]

    expected = sum(estimate_message_tokens(message) for message in messages) + 2
    assert estimate_prompt_tokens(messages) == expected


def test_trim_keeps_system_message_and_newest_complete_messages():
    messages = [
        ChatMessage(role="system", content="rules"),
        ChatMessage(role="user", content="old request"),
        ChatMessage(role="assistant", content="old response"),
        ChatMessage(role="user", content="latest request"),
    ]
    budget = (
        2
        + estimate_message_tokens(messages[0])
        + estimate_message_tokens(messages[3])
    )

    assert trim_messages(messages, budget) == [messages[0], messages[3]]


def test_trim_does_not_include_partial_messages():
    messages = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="this message is too large"),
        ChatMessage(role="user", content="latest"),
    ]
    budget = 2 + estimate_message_tokens(messages[2])

    assert trim_messages(messages, budget) == [messages[2]]


def test_trim_rejects_budget_smaller_than_reply_primer():
    with pytest.raises(ValueError, match="at least 2"):
        trim_messages([], max_prompt_tokens=1)


def test_trim_rejects_system_message_that_cannot_fit():
    system_message = ChatMessage(role="system", content="rules")

    with pytest.raises(ValueError, match="system message"):
        trim_messages([system_message], max_prompt_tokens=2)


def test_estimation_handles_multibyte_utf8_text():
    message = ChatMessage(role="user", content="世界")

    assert estimate_message_tokens(message) == len(b"user") + len("世界".encode("utf-8")) + 4
