"""Tests for Stateful Conversation & Memory Architecture."""

import pytest

from src.agent.memory import (
    CompositeMemory,
    EntityMemory,
    Message,
    SummaryMemory,
    TokenBufferMemory,
)


class TestMessage:
    def test_message_creation_and_auto_token_estimation(self):
        msg = Message(role="user", content="Hello, tell me about Python.")
        assert msg.role == "user"
        assert msg.content == "Hello, tell me about Python."
        assert msg.token_count > 0
        assert msg.to_dict() == {"role": "user", "content": "Hello, tell me about Python."}

    def test_explicit_token_count_override(self):
        msg = Message(role="assistant", content="Hi there!", token_count=10)
        assert msg.token_count == 10


class TestTokenBufferMemory:
    def test_add_and_retrieve_messages(self):
        memory = TokenBufferMemory(max_tokens=100)
        memory.add_message("user", "Hello world")
        memory.add_message("assistant", "Hi! How can I help?")

        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_pinned_system_prompt_never_evicted(self):
        # Budget fits system prompt (approx 5 tokens) + 1 message (approx 5 tokens), total budget = 12
        memory = TokenBufferMemory(max_tokens=12, system_prompt="You are a helpful assistant.")
        memory.add_message("user", "Turn 1: Alpha message here")
        memory.add_message("assistant", "Turn 1: Beta reply here")
        memory.add_message("user", "Turn 2: Gamma message here")

        messages = memory.get_messages()
        # First message MUST be the system prompt
        assert messages[0].role == "system"
        assert messages[0].content == "You are a helpful assistant."
        # Total tokens must not exceed max_tokens
        total_tokens = sum(m.token_count for m in messages)
        assert total_tokens <= memory.max_tokens

    def test_fifo_eviction_under_tight_budget(self):
        memory = TokenBufferMemory(max_tokens=20)
        memory.add_message("user", "Oldest message that should be evicted first", token_count=10)
        memory.add_message("assistant", "Middle message", token_count=8)
        memory.add_message("user", "Newest message", token_count=8)

        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Middle message"
        assert messages[1].content == "Newest message"

    def test_clear_preserves_system_prompt(self):
        memory = TokenBufferMemory(max_tokens=100, system_prompt="Pinned system prompt")
        memory.add_message("user", "Temporary user message")
        memory.clear()

        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content == "Pinned system prompt"


class TestSummaryMemory:
    def test_summarization_trigger_on_overflow(self):
        # Mock summarizer function
        def mock_summarizer(existing_summary: str, messages: list[Message]) -> str:
            msg_texts = "; ".join(f"{m.role}: {m.content}" for m in messages)
            if existing_summary:
                return f"{existing_summary} | {msg_texts}"
            return f"Summary: {msg_texts}"

        memory = SummaryMemory(
            max_unsummarized_tokens=20,
            summarizer_fn=mock_summarizer,
        )

        # Add messages that exceed token limit
        memory.add_message("user", "My name is Alice and I work in Chicago.", token_count=12)
        memory.add_message("assistant", "Nice to meet you Alice!", token_count=12)

        # Triggers compaction
        summary = memory.get_summary()
        assert "Alice" in summary
        assert "Chicago" in summary

    def test_get_context_combines_summary_and_recent_buffer(self):
        def dummy_summarizer(existing: str, msgs: list[Message]) -> str:
            return "User is an engineer."

        memory = SummaryMemory(
            max_unsummarized_tokens=15,
            summarizer_fn=dummy_summarizer,
        )
        memory.add_message("user", "I am a software engineer.", token_count=10)
        memory.add_message("assistant", "Great to know!", token_count=10)
        memory.add_message("user", "What framework should I use?", token_count=5)

        context_messages = memory.get_messages()
        # Should have a system summary message + the remaining recent message
        assert any("User is an engineer" in m.content for m in context_messages)
        assert context_messages[-1].content == "What framework should I use?"


class TestEntityMemory:
    def test_extract_and_store_entities(self):
        entity_mem = EntityMemory()
        entity_mem.add_entity_fact("Alice", "role", "Lead AI Engineer")
        entity_mem.add_entity_fact("Alice", "location", "San Francisco")
        entity_mem.add_entity_fact("Project Raven", "status", "active")

        facts = entity_mem.get_entity_facts("Alice")
        assert facts["role"] == "Lead AI Engineer"
        assert facts["location"] == "San Francisco"

    def test_extract_entities_from_text(self):
        entity_mem = EntityMemory()
        # Rule-based / pattern entity extraction: "key is value", "key: value"
        text = "Preference: dark_mode. Language: Python."
        entity_mem.extract_from_text(text)

        assert entity_mem.get_fact("Preference") == "dark_mode"
        assert entity_mem.get_fact("Language") == "Python"

    def test_query_relevant_entity_context(self):
        entity_mem = EntityMemory()
        entity_mem.add_entity_fact("Database", "type", "PostgreSQL")
        entity_mem.add_entity_fact("Frontend", "framework", "React")

        context_str = entity_mem.get_relevant_context("Tell me about our Database setup")
        assert "Database" in context_str
        assert "PostgreSQL" in context_str
        assert "React" not in context_str


class TestCompositeMemory:
    def test_unified_context_assembly(self):
        def dummy_summarizer(existing: str, msgs: list[Message]) -> str:
            return "User asked for Python assistance."

        composite = CompositeMemory(
            max_tokens=60,
            system_prompt="You are an autonomous AI assistant.",
            summarizer_fn=dummy_summarizer,
        )
        composite.add_entity_fact("User", "preferred_language", "Python")

        composite.add_message("user", "Hello! Can you help me write a function?", token_count=10)
        composite.add_message("assistant", "Certainly! What function do you need?", token_count=10)

        payload = composite.get_chat_history(current_query="Help me with Python code")

        # System prompt should be at index 0
        assert payload[0]["role"] == "system"
        assert "autonomous AI assistant" in payload[0]["content"]

        # Entity memory should be injected either in system message or dedicated context message
        all_content = " ".join(m["content"] for m in payload)
        assert "preferred_language" in all_content or "Python" in all_content

        # Recent messages preserved
        assert any("Certainly!" in m["content"] for m in payload)
