"""Database access: connection helpers and the pgvector-backed store."""

from __future__ import annotations

import json
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from rag.config import settings


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def insert_chunks(rows: list[dict[str, Any]]) -> int:
    """Insert chunk rows. Each row needs source, chunk_index, content, metadata, embedding."""
    if not rows:
        return 0

    sql = """
        INSERT INTO documents (source, chunk_index, content, metadata, embedding)
        VALUES (%(source)s, %(chunk_index)s, %(content)s, %(metadata)s::jsonb, %(embedding)s)
    """
    prepared = [
        {
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "metadata": json.dumps(r.get("metadata") or {}),
            "embedding": Vector(r["embedding"]),
        }
        for r in rows
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, prepared)
        conn.commit()
    return len(prepared)


def search(query_embedding: list[float], k: int) -> list[dict[str, Any]]:
    """Top-k cosine similarity search. Returns rows with ascending distance."""
    query = Vector(query_embedding)
    sql = """
        SELECT content, source, metadata, embedding <=> %s AS distance
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (query, query, k))
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def clear_documents() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE documents RESTART IDENTITY")
        conn.commit()
