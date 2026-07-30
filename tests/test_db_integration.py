"""Integration tests for the pgvector-backed store, against a real Postgres."""

import pytest

from rag.config import settings

pytestmark = pytest.mark.integration


def _embedding(value: float) -> list[float]:
    return [value] * settings.embedding_dim


def _row(source: str, chunk_index: int, content: str, *, content_hash: str) -> dict:
    return {
        "source": source,
        "chunk_index": chunk_index,
        "content": content,
        "content_hash": content_hash,
        "metadata": {"title": "Guide", "file_hash": "file-1"},
        "embedding": _embedding(0.1 + chunk_index / 100),
    }


def test_schema_applies_and_upsert_stores_a_chunk(empty_index):
    db = empty_index

    db.upsert_chunks(
        [_row("docs/guide.md", 0, "BRIN stores block range summaries.", content_hash="h0")]
    )

    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT source, chunk_index, content_hash FROM documents")
        assert cur.fetchall() == [("docs/guide.md", 0, "h0")]


def test_generated_search_vector_is_populated(empty_index):
    """The tsvector column is GENERATED ALWAYS, so it must fill itself on insert."""
    db = empty_index

    db.upsert_chunks(
        [_row("docs/guide.md", 0, "BRIN stores block range summaries.", content_hash="h0")]
    )

    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT search_vector @@ plainto_tsquery('english', 'brin') FROM documents")
        assert cur.fetchone()[0] is True


def test_metadata_round_trips_as_jsonb(empty_index):
    db = empty_index

    db.upsert_chunks([_row("docs/guide.md", 0, "Content here.", content_hash="h0")])

    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT metadata->>'title', metadata->>'file_hash' FROM documents")
        assert cur.fetchone() == ("Guide", "file-1")


def test_source_file_hash_reads_back_the_stored_hash(empty_index):
    db = empty_index

    db.upsert_chunks([_row("docs/guide.md", 0, "Content here.", content_hash="h0")])

    assert db.source_file_hash("docs/guide.md") == "file-1"
    assert db.source_file_hash("docs/missing.md") is None
