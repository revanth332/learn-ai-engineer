"""Document chunking strategies from first principles."""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    """Split text into fixed-size character chunks with specified overlap.

    Sliding window advances by: step = chunk_size - chunk_overlap.

    Args:
        text: Raw text string to be chunked.
        chunk_size: Maximum number of characters in each chunk (> 0).
        chunk_overlap: Number of characters to overlap between adjacent chunks (0 <= chunk_overlap < chunk_size).

    Returns:
        List of chunk strings. Returns empty list if text is empty.

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size.
    """
    if text == "":
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
    chunks = []
    step = chunk_size - chunk_overlap
    for i in range(0,len(text),step):
        chunks.append(text[i:i+chunk_size])
        if i+chunk_size >= len(text):
            break
    return chunks

