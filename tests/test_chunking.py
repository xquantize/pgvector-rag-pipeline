"""Unit tests for chunking."""

from rag.chunking import chunk_text


def test_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_tiny_input_single_chunk():
    text = "hello world"
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert chunks == ["hello world"]


def test_respects_chunk_size_and_overlap():
    # Build text that is definitely longer than chunk_size tokens.
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    size, overlap = 40, 10
    chunks = chunk_text(text, chunk_size=size, overlap=overlap)
    assert len(chunks) > 1

    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    token_lens = [len(enc.encode(c)) for c in chunks]
    # All but possibly the last chunk should be <= size; last may be shorter.
    assert all(n <= size for n in token_lens)
    assert token_lens[0] == size


def test_overlap_windows_match_token_slices():
    import tiktoken

    text = " ".join(f"tok{i}" for i in range(100))
    size, overlap = 30, 10
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = chunk_text(text, chunk_size=size, overlap=overlap)

    assert len(chunks) >= 2
    step = size - overlap
    assert chunks[0] == enc.decode(tokens[0:size]).strip()
    assert chunks[1] == enc.decode(tokens[step : step + size]).strip()
