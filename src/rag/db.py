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


def source_file_hash(source: str) -> str | None:
    """Return stored file_hash for a source, if any chunks exist."""
    sql = """
        SELECT metadata->>'file_hash'
        FROM documents
        WHERE source = %s
        LIMIT 1
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (source,))
        row = cur.fetchone()
        return row[0] if row else None


def delete_source(source: str) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE source = %s", (source,))
        conn.commit()


def delete_stale_chunks(source: str, keep_count: int) -> None:
    """Remove leftover chunks after a file shrinks."""
    sql = "DELETE FROM documents WHERE source = %s AND chunk_index >= %s"
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (source, keep_count))
        conn.commit()


def upsert_chunks(rows: list[dict[str, Any]]) -> int:
    """Idempotent upsert on (source, chunk_index). Skips embed-identical rows via hash."""
    if not rows:
        return 0

    sql = """
        INSERT INTO documents (source, chunk_index, content, content_hash, metadata, embedding)
        VALUES (
            %(source)s, %(chunk_index)s, %(content)s, %(content_hash)s,
            %(metadata)s::jsonb, %(embedding)s
        )
        ON CONFLICT (source, chunk_index) DO UPDATE SET
            content = EXCLUDED.content,
            content_hash = EXCLUDED.content_hash,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding
        WHERE documents.content_hash IS DISTINCT FROM EXCLUDED.content_hash
    """
    prepared = [
        {
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "content_hash": r["content_hash"],
            "metadata": json.dumps(r.get("metadata") or {}),
            "embedding": Vector(r["embedding"]),
        }
        for r in rows
    ]
    with get_connection() as conn, conn.cursor() as cur:
        cur.executemany(sql, prepared)
        conn.commit()
    return len(prepared)


def search(
    query_embedding: list[float],
    k: int,
    *,
    query_text: str | None = None,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """Top-k search. mode=vector|hybrid (default from settings)."""
    retrieval_mode = mode or settings.retrieval_mode
    query = Vector(query_embedding)

    if retrieval_mode == "hybrid" and query_text:
        # Reciprocal Rank Fusion over vector ANN and FTS in SQL.
        sql = """
            WITH vector_hits AS (
                SELECT id, content, source, metadata,
                       embedding <=> %s AS distance,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> %s) AS rank
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT %s
            ),
            fts_hits AS (
                SELECT id, content, source, metadata,
                       embedding <=> %s AS distance,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC
                       ) AS rank
                FROM documents
                WHERE search_vector @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC
                LIMIT %s
            ),
            fused AS (
                SELECT
                    COALESCE(v.id, f.id) AS id,
                    COALESCE(v.content, f.content) AS content,
                    COALESCE(v.source, f.source) AS source,
                    COALESCE(v.metadata, f.metadata) AS metadata,
                    COALESCE(v.distance, f.distance) AS distance,
                    COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + f.rank), 0) AS score
                FROM vector_hits v
                FULL OUTER JOIN fts_hits f ON v.id = f.id
            )
            SELECT content, source, metadata, distance, score
            FROM fused
            ORDER BY score DESC, distance ASC
            LIMIT %s
        """
        fts_limit = max(k * 4, 20)
        params = (
            query,
            query,
            query,
            fts_limit,
            query,
            query_text,
            query_text,
            query_text,
            fts_limit,
            k,
        )
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

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
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE documents RESTART IDENTITY")
        conn.commit()
