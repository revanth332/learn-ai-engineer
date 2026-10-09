"""Stateful Conversation & Memory Architecture.

This module implements production-grade conversational memory:
- Pinned System Prompts with FIFO Token Budget Eviction (TokenBufferMemory)
- Recursive Rolling Summarization (SummaryMemory)
- Relational / Key-Value Entity Memory (EntityMemory)
- Unified Composite Memory Controller (CompositeMemory)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Message:
    """Represents a single chat message in memory."""

    role: str
    content: str
    token_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            # Deterministic character-based token estimation (~4 chars per token, minimum 1)
            self.token_count = max(1, len(self.content) // 4)

    def to_dict(self) -> dict[str, str]:
        """Convert message to standard OpenAI chat completion dict format."""
        return {"role": self.role, "content": self.content}


class TokenBufferMemory:
    """Sliding-window conversational buffer with pinned system prompt and strict token budgeting."""

    def __init__(self, max_tokens: int = 2000, system_prompt: str | None = None) -> None:
        self.max_tokens = max_tokens
        self.system_prompt: Message | None = (
            Message(role="system", content=system_prompt) if system_prompt else None
        )
        self.buffer: list[Message] = []

    def _get_system_tokens(self) -> int:
        return self.system_prompt.token_count if self.system_prompt else 0

    def add_message(self, role: str, content: str, token_count: int = 0) -> None:
        """Add a new message to the buffer and evict oldest non-system messages if over budget."""
        message = Message(role=role, content=content, token_count=token_count)
        self.buffer.append(message)

        # Evict oldest non-system messages until total token usage is within budget
        allowed_dynamic_tokens = self.max_tokens - self._get_system_tokens()
        current_tokens = sum(m.token_count for m in self.buffer)

        while self.buffer and current_tokens > allowed_dynamic_tokens:
            evicted = self.buffer.pop(0)
            current_tokens -= evicted.token_count

    def get_messages(self) -> list[Message]:
        """Return the current list of messages within the token budget."""
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(self.system_prompt)
        messages.extend(self.buffer)
        return messages

    def clear(self) -> None:
        """Clear dynamic message history while preserving pinned system prompt."""
        self.buffer.clear()


class SummaryMemory:
    """Rolling summarizer memory that compresses older turns when buffer exceeds token threshold."""

    def __init__(
        self,
        max_unsummarized_tokens: int = 500,
        summarizer_fn: Callable[[str, list[Message]], str] | None = None,
    ) -> None:
        self.max_unsummarized_tokens = max_unsummarized_tokens
        self.summarizer_fn = summarizer_fn or self._default_summarizer
        self.summary: str = ""
        self.unsummarized_buffer: list[Message] = []

    @staticmethod
    def _default_summarizer(existing_summary: str, messages: list[Message]) -> str:
        new_text = " | ".join(f"{m.role}: {m.content}" for m in messages)
        if existing_summary:
            return f"{existing_summary} | {new_text}"
        return f"Summary: {new_text}"

    def add_message(self, role: str, content: str, token_count: int = 0) -> None:
        """Add message to unsummarized buffer and trigger summarizer if threshold exceeded."""
        message = Message(role=role, content=content, token_count=token_count)
        self.unsummarized_buffer.append(message)

        total_tokens = sum(m.token_count for m in self.unsummarized_buffer)
        if total_tokens > self.max_unsummarized_tokens:
            self.summary = self.summarizer_fn(self.summary, self.unsummarized_buffer)
            self.unsummarized_buffer.clear()

    def get_summary(self) -> str:
        """Return current cumulative summary."""
        return self.summary

    def get_messages(self) -> list[Message]:
        """Return summary message (if any) followed by unsummarized messages."""
        messages: list[Message] = []
        if self.summary:
            messages.append(Message(role="system", content=f"Previous conversation summary: {self.summary}"))
        messages.extend(self.unsummarized_buffer)
        return messages


class EntityMemory:
    """Entity-relationship memory store that extracts and injects structured entity facts."""

    def __init__(self) -> None:
        # Stores entity -> {attribute: value}
        self.entities: dict[str, dict[str, Any]] = {}

    def add_entity_fact(self, entity: str, attribute: str, value: Any) -> None:
        """Store an entity attribute and value."""
        if entity not in self.entities:
            self.entities[entity] = {}
        self.entities[entity][attribute] = value

    def get_entity_facts(self, entity: str) -> dict[str, Any]:
        """Retrieve all recorded facts for a specific entity."""
        return self.entities.get(entity, {})

    def get_fact(self, entity_or_key: str) -> Any:
        """Retrieve a specific fact by entity name or single flat key."""
        if entity_or_key in self.entities:
            facts = self.entities[entity_or_key]
            if len(facts) == 1:
                return next(iter(facts.values()))
            return facts
        return None

    def extract_from_text(self, text: str) -> None:
        """Extract 'Key: Value' or 'Entity attribute is Value' patterns from text."""
        # Pattern 1: "Key: Value" or "Key = Value"
        colon_matches = re.findall(r"([A-Za-z0-9_\s]+)\s*[:=]\s*([^\n,.;]+)", text)
        for key, val in colon_matches:
            k = key.strip()
            v = val.strip()
            if k and v:
                self.add_entity_fact(k, "value", v)

        # Pattern 2: "Entity's attribute is Value" or "Entity attribute is Value"
        is_matches = re.findall(r"([A-Za-z0-9_]+)(?:'s)?\s+([A-Za-z0-9_]+)\s+is\s+([^\n,.;]+)", text, re.IGNORECASE)
        for entity, attr, val in is_matches:
            e = entity.strip()
            a = attr.strip()
            v = val.strip()
            if e and a and v:
                self.add_entity_fact(e, a, v)

    def get_relevant_context(self, query: str) -> str:
        """Return formatted string of entity facts matching keywords in query."""
        if not query or not self.entities:
            return ""

        query_lower = query.lower()
        matched_facts: list[str] = []

        for entity, attributes in self.entities.items():
            if entity.lower() in query_lower:
                attr_str = ", ".join(f"{k}: {v}" for k, v in attributes.items())
                matched_facts.append(f"{entity} ({attr_str})")

        if not matched_facts:
            return ""
        return "Relevant Entity Context: " + "; ".join(matched_facts)


class CompositeMemory:
    """Orchestrates pinned system prompts, entity context injection, and rolling/sliding buffers."""

    def __init__(
        self,
        max_tokens: int = 2000,
        system_prompt: str = "You are a helpful assistant.",
        summarizer_fn: Callable[[str, list[Message]], str] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.token_buffer = TokenBufferMemory(max_tokens=max_tokens, system_prompt=system_prompt)
        self.summary_memory = SummaryMemory(
            max_unsummarized_tokens=max_tokens // 2, summarizer_fn=summarizer_fn
        )
        self.entity_memory = EntityMemory()

    def add_message(self, role: str, content: str, token_count: int = 0) -> None:
        """Ingest turn across memory components and extract entity facts."""
        self.token_buffer.add_message(role, content, token_count=token_count)
        self.summary_memory.add_message(role, content, token_count=token_count)
        if role == "user":
            self.entity_memory.extract_from_text(content)

    def add_entity_fact(self, entity: str, attribute: str, value: Any) -> None:
        """Manually register an entity fact."""
        self.entity_memory.add_entity_fact(entity, attribute, value)

    def get_chat_history(self, current_query: str = "") -> list[dict[str, str]]:
        """Assemble full chat completion payload."""
        messages: list[dict[str, str]] = []

        # 1. Base system prompt
        system_content = self.system_prompt

        # 2. Extract and append relevant entity facts if present
        entity_context = self.entity_memory.get_relevant_context(current_query)
        if entity_context:
            system_content = f"{system_content}\n\n{entity_context}"

        messages.append({"role": "system", "content": system_content})

        # 3. Add dynamic conversational buffer
        for msg in self.token_buffer.buffer:
            messages.append(msg.to_dict())

        return messages
