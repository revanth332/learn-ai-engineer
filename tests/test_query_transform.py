"""Unit tests for Query Transformation module."""

import pytest
from src.retrieval.query_transform import (
    expand_query_synonyms,
    decompose_query,
    generate_hyde_prompt,
    fuse_transformed_queries,
)


# --- Tests for expand_query_synonyms ---

def test_expand_query_synonyms_single_match():
    query = "fast database"
    synonym_map = {"fast": ["quick", "low-latency"], "database": ["datastore"]}
    expanded = expand_query_synonyms(query, synonym_map)
    # Must preserve original query and append new synonyms without duplicates
    tokens = expanded.lower().split()
    assert "fast" in tokens
    assert "database" in tokens
    assert "quick" in tokens
    assert "low-latency" in tokens
    assert "datastore" in tokens


def test_expand_query_synonyms_no_match():
    query = "kubernetes pod"
    synonym_map = {"redis": ["in-memory store"]}
    expanded = expand_query_synonyms(query, synonym_map)
    assert expanded.strip() == "kubernetes pod"


def test_expand_query_synonyms_avoids_duplicate_injection():
    query = "quick quick runner"
    synonym_map = {"runner": ["sprinter", "athlete"]}
    expanded = expand_query_synonyms(query, synonym_map)
    tokens = expanded.lower().split()
    # "sprinter" and "athlete" should each only appear once
    assert tokens.count("sprinter") == 1
    assert tokens.count("athlete") == 1


# --- Tests for decompose_query ---

def test_decompose_query_conjunctions():
    query = "Postgres latency and Redis throughput, and Kafka streaming"
    sub_queries = decompose_query(query)
    assert len(sub_queries) == 3
    assert sub_queries[0] == "Postgres latency"
    assert sub_queries[1] == "Redis throughput"
    assert sub_queries[2] == "Kafka streaming"


def test_decompose_query_comparisons():
    query = "PyTorch vs TensorFlow versus JAX"
    sub_queries = decompose_query(query)
    assert len(sub_queries) == 3
    assert sub_queries[0] == "PyTorch"
    assert sub_queries[1] == "TensorFlow"
    assert sub_queries[2] == "JAX"


def test_decompose_query_single_clause():
    query = "What is a vector index?"
    sub_queries = decompose_query(query)
    assert sub_queries == ["What is a vector index?"]


def test_decompose_query_empty_or_whitespace():
    assert decompose_query("") == []
    assert decompose_query("   ") == []


# --- Tests for generate_hyde_prompt ---

def test_generate_hyde_prompt_general():
    prompt = generate_hyde_prompt("What is gradient descent?")
    assert "Please write a short, highly technical passage that answers the following general question." in prompt
    assert "Question: What is gradient descent?" in prompt
    assert prompt.endswith("Passage:")


def test_generate_hyde_prompt_custom_domain():
    prompt = generate_hyde_prompt("How does mRNA vaccine work?", domain="medical")
    assert "Please write a short, highly technical passage that answers the following medical question." in prompt
    assert "Question: How does mRNA vaccine work?" in prompt
    assert prompt.endswith("Passage:")


# --- Tests for fuse_transformed_queries ---

def test_fuse_transformed_queries_ordering_and_deduplication():
    original = "vector database"
    expanded = "vector database embedding store index"
    sub_queries = ["vector database", "embedding store", "approximate nearest neighbor"]

    fused = fuse_transformed_queries(original, expanded, sub_queries)
    assert fused[0] == "vector database"
    assert fused[1] == "vector database embedding store index"
    assert "embedding store" in fused
    assert "approximate nearest neighbor" in fused
    assert len(fused) == 4  # Duplicate 'vector database' in sub_queries eliminated


def test_fuse_transformed_queries_none_inputs():
    fused = fuse_transformed_queries("standalone query", expanded_query=None, sub_queries=None)
    assert fused == ["standalone query"]


def test_fuse_transformed_queries_empty_original():
    fused = fuse_transformed_queries("", expanded_query="expanded text", sub_queries=["sub 1"])
    assert fused == ["expanded text", "sub 1"]
