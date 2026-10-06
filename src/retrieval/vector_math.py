"""Vector math operations from first principles."""

import math


def dot_product(u: list[float], v: list[float]) -> float:
    """Compute the dot product of two vectors: sum(u[i] * v[i]).

    Args:
        u: First vector of float values.
        v: Second vector of float values.

    Returns:
        Scalar dot product.

    Raises:
        ValueError: If vectors have different lengths or are empty.
    """
    if u == [] or v == [] or len(u) != len(v):
        raise ValueError("Invalid vectors")

    return sum([u[i] * v[i] for i in range(len(u))])


def norm(v: list[float]) -> float:
    """Compute the L2 Euclidean norm (magnitude) of a vector: sqrt(sum(v[i]^2)).

    Args:
        v: Vector of float values.

    Returns:
        Scalar L2 norm.

    Raises:
        ValueError: If vector is empty.
    """
    if v == []:
        raise ValueError("Empty vector")
    return pow(sum([vi**2 for vi in v]),0.5)


def cosine_similarity(u: list[float], v: list[float]) -> float:
    """Compute the cosine similarity between two vectors: (u . v) / (||u|| * ||v||).

    Cosine similarity ranges from -1.0 to 1.0 (or 0.0 to 1.0 for non-negative embeddings).
    If either vector has norm 0.0, return 0.0.

    Args:
        u: First vector.
        v: Second vector.

    Returns:
        Cosine similarity score as float.

    Raises:
        ValueError: If vectors have different lengths or are empty.
    """
    if u == [] or v == [] or len(u) != len(v):
        raise ValueError("Invalid vectors")
    norm_u = norm(u)
    norm_v = norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot_product(u,v)/(norm(u) * norm(v))


def euclidean_distance(u: list[float], v: list[float]) -> float:
    """Compute the Euclidean distance (L2 distance) between two vectors.

    Formula: sqrt(sum((u[i] - v[i])^2))

    Args:
        u: First vector.
        v: Second vector.

    Returns:
        Euclidean distance as float.

    Raises:
        ValueError: If vectors have different lengths or are empty.
    """
    if u == [] or v == [] or len(u) != len(v):
        raise ValueError("Invalid vectors")
    return pow(sum([(u[i]-v[i])**2 for i in range(len(u))]),0.5)


def normalize(v: list[float]) -> list[float]:
    """Normalize a vector to have unit length (L2 norm = 1.0).

    Args:
        v: Vector to normalize.

    Returns:
        Unit vector with same direction. If zero-vector, returns copy of vector.

    Raises:
        ValueError: If vector is empty.
    """
    if v == []:
        raise ValueError("Empty vector")
    l2_norm = norm(v)
    if l2_norm == 0:
        return v
    return [d/l2_norm for d in v]