from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ChatMessage:
    """A chat message represented in the format consumed by a chat model."""

    role: str
    content: str


def estimate_message_tokens(message: ChatMessage) -> int:
    """Estimate tokens using UTF-8 bytes plus fixed message framing.

    The extra four represents the role/content separators and message
    delimiters used by the chat format; it is a heuristic, not a tokenizer
    result.
    """
    return len(message.role.encode("utf-8")) + len(message.content.encode("utf-8")) + 4


def estimate_prompt_tokens(messages: Sequence[ChatMessage]) -> int:
    """Estimate total prompt tokens, including the assistant reply primer."""
    return sum(estimate_message_tokens(message) for message in messages) + 2


def trim_messages(
    messages: Sequence[ChatMessage],
    max_prompt_tokens: int,
) -> list[ChatMessage]:
    """Keep the system message and newest complete messages within a token budget."""
    if max_prompt_tokens < 2:
        raise ValueError("max_prompt_tokens must be at least 2")

    system_message = messages[0] if messages and messages[0].role == "system" else None
    remaining_messages = messages[1:] if system_message else messages
    selected: list[ChatMessage] = []
    used_tokens = 2

    if system_message:
        used_tokens += estimate_message_tokens(system_message)
        if used_tokens > max_prompt_tokens:
            raise ValueError("system message exceeds max_prompt_tokens")

    for message in reversed(remaining_messages):
        message_tokens = estimate_message_tokens(message)
        if used_tokens + message_tokens > max_prompt_tokens:
            break
        selected.append(message)
        used_tokens += message_tokens

    result = ([system_message] if system_message else []) + list(reversed(selected))
    return result
