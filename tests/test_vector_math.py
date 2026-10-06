import math
import pytest
from src.retrieval.vector_math import (
    dot_product,
    norm,
    cosine_similarity,
    euclidean_distance,
    normalize,
)


def test_dot_product_basic():
    u = [1.0, 2.0, 3.0]
    v = [4.0, 5.0, 6.0]
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert dot_product(u, v) == 32.0


def test_dot_product_orthogonal():
    u = [1.0, 0.0]
    v = [0.0, 1.0]
    assert dot_product(u, v) == 0.0


def test_dot_product_length_mismatch_and_empty():
    with pytest.raises(ValueError):
        dot_product([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        dot_product([], [])


def test_norm_basic():
    v = [3.0, 4.0]
    assert norm(v) == 5.0

    v3d = [1.0, 2.0, 2.0]
    assert norm(v3d) == 3.0


def test_norm_empty():
    with pytest.raises(ValueError):
        norm([])


def test_cosine_similarity_identical():
    u = [1.0, 2.0, 3.0]
    assert pytest.approx(cosine_similarity(u, u), rel=1e-5) == 1.0


def test_cosine_similarity_opposite():
    u = [1.0, 0.0]
    v = [-1.0, 0.0]
    assert pytest.approx(cosine_similarity(u, v), rel=1e-5) == -1.0


def test_cosine_similarity_orthogonal():
    u = [1.0, 0.0]
    v = [0.0, 1.0]
    assert pytest.approx(cosine_similarity(u, v), rel=1e-5) == 0.0


def test_cosine_similarity_zero_vector():
    u = [0.0, 0.0]
    v = [1.0, 2.0]
    assert cosine_similarity(u, v) == 0.0


def test_cosine_similarity_validation():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_euclidean_distance_basic():
    u = [0.0, 0.0]
    v = [3.0, 4.0]
    assert euclidean_distance(u, v) == 5.0


def test_euclidean_distance_identical():
    u = [1.0, 2.0, 3.0]
    assert euclidean_distance(u, u) == 0.0


def test_euclidean_distance_validation():
    with pytest.raises(ValueError):
        euclidean_distance([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        euclidean_distance([], [])


def test_normalize_basic():
    v = [3.0, 4.0]
    unit_v = normalize(v)
    assert pytest.approx(norm(unit_v), rel=1e-5) == 1.0
    assert pytest.approx(unit_v[0], rel=1e-5) == 0.6
    assert pytest.approx(unit_v[1], rel=1e-5) == 0.8


def test_normalize_zero_vector():
    zero_v = [0.0, 0.0, 0.0]
    assert normalize(zero_v) == [0.0, 0.0, 0.0]


def test_normalize_empty():
    with pytest.raises(ValueError):
        normalize([])
