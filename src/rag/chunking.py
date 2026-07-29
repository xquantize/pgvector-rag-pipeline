"""Chunking strategies — token-aware fixed-size splitter."""

from __future__ import annotations

import tiktoken

from rag.config import settings

_ENCODING_NAME = "cl100k_base"


def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping token windows of `chunk_size`."""
    size = chunk_size if chunk_size is not None else settings.chunk_size
    ov = overlap if overlap is not None else settings.chunk_overlap
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if ov < 0 or ov >= size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    cleaned = (text or "").strip()
    if not cleaned:
        return []

    enc = _encoder()
    tokens = enc.encode(cleaned)
    if len(tokens) <= size:
        return [cleaned]

    step = size - ov
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        chunks.append(enc.decode(window).strip())
        if start + size >= len(tokens):
            break
    return [c for c in chunks if c]
