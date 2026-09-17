from src.tokenization.streaming import iter_sse_data


def test_yields_data_from_complete_sse_events():
    chunks = [
        b"data: first token\n\n",
        b"data: second token\n\n",
    ]

    assert list(iter_sse_data(chunks)) == ["first token", "second token"]


def test_reassembles_events_split_across_byte_chunks():
    chunks = [
        b"data: hel",
        b"lo\n\n",
        b"data: wor",
        b"ld\n\n",
    ]

    assert list(iter_sse_data(chunks)) == ["hello", "world"]


def test_yields_multiple_events_from_one_byte_chunk():
    chunks = [b"data: one\n\ndata: two\n\n"]

    assert list(iter_sse_data(chunks)) == ["one", "two"]


def test_preserves_an_incomplete_event_after_a_complete_one():
    chunks = [
        b"data: first\n\ndata: sec",
        b"ond\n\n",
    ]

    assert list(iter_sse_data(chunks)) == ["first", "second"]


def test_ignores_non_data_sse_lines():
    chunks = [
        b": keep-alive\n\n",
        b"event: message\ndata: useful\nid: 42\n\n",
    ]

    assert list(iter_sse_data(chunks)) == ["useful"]


def test_done_sentinel_ends_the_stream():
    chunks = [
        b"data: first\n\n",
        b"data: [DONE]\n\n",
        b"data: must not be emitted\n\n",
    ]

    assert list(iter_sse_data(chunks)) == ["first"]


def test_empty_stream_yields_no_events():
    assert list(iter_sse_data([])) == []
