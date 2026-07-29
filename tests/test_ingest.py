"""Tests for ingestion lifecycle behavior that does not require Postgres."""

from rag import ingest


def test_ingest_directory_removes_sources_for_deleted_files(tmp_path, monkeypatch):
    current = tmp_path / "current.md"
    current.write_text("current", encoding="utf-8")
    stale = (tmp_path / "deleted.md").as_posix()
    unrelated = (tmp_path.parent / "other" / "document.md").as_posix()
    deleted: list[str] = []

    monkeypatch.setattr(ingest.db, "list_sources", lambda: [stale, unrelated])
    monkeypatch.setattr(ingest.db, "delete_source", deleted.append)
    monkeypatch.setattr(
        ingest.db,
        "source_file_hash",
        lambda _source: ingest._sha256("current"),
    )

    assert ingest.ingest_path(tmp_path) == 0
    assert deleted == [stale]
