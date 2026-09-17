"""Utilities for decoding streamed Server-Sent Events (SSE) responses."""

from collections.abc import Iterable, Iterator


def iter_sse_data(chunks: Iterable[bytes]) -> Iterator[str]:
    """Yield `data:` payloads from a byte stream of SSE events.

    Events are separated by a blank line. The `[DONE]` payload ends the stream.
    """
    buffer = ""
    for chunk in chunks:
        chunk_str = chunk.decode('utf-8')
        buffer += chunk_str

        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n",1)
            props = event.split("\n")
            for prop in props:
                index = prop.find("data: ")
                if index != -1:
                    content = prop[index+6:]
                    if content == "[DONE]":
                        return
                    yield content

# chunks = [
#         b"data: hel",
#         b"lo\n\n",
#         b"data: wor",
#         b"ld\n\n",
#     ]

# print(iter_sse_data(chunks))
# chunks = "data: one\n\ndata: two\n\n"
# print(chunks.split("\n\n"))