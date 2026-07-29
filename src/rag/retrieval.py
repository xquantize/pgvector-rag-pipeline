"""Retrieval: turn a question into the most relevant chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag import db
from rag.config import settings
from rag.embeddings import embed_query


@dataclass
class Chunk:
    content: str
    source: str
    metadata: dict[str, Any]
    distance: float


def retrieve(question: str, k: int | None = None) -> list[Chunk]:
    top_k = k if k is not None else settings.top_k
    embedding = embed_query(question)
    rows = db.search(embedding, top_k)
    results: list[Chunk] = []
    for row in rows:
        meta = row.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = dict(meta)
        results.append(
            Chunk(
                content=row["content"],
                source=row["source"],
                metadata=meta,
                distance=float(row["distance"]),
            )
        )
    return results
