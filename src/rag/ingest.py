"""Ingestion pipeline: documents -> chunks -> embeddings -> pgvector."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

from rag import db
from rag.chunking import chunk_text
from rag.embeddings import embed_texts

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    raise ValueError(f"Unsupported file type: {path}")


def iter_documents(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def ingest_path(path: str | Path, *, clear: bool = False) -> int:
    """Load files under `path`, chunk, embed, and store. Returns chunk count."""
    root = Path(path)
    files = iter_documents(root)
    if not files:
        raise FileNotFoundError(f"No supported documents found under {root}")

    if clear:
        db.clear_documents()

    rows: list[dict] = []
    for file_path in tqdm(files, desc="chunking"):
        text = load_file(file_path)
        chunks = chunk_text(text)
        for idx, content in enumerate(chunks):
            rows.append(
                {
                    "source": str(file_path.as_posix()),
                    "chunk_index": idx,
                    "content": content,
                    "metadata": {"filename": file_path.name},
                }
            )

    if not rows:
        return 0

    texts = [r["content"] for r in rows]
    embeddings = embed_texts(texts)
    for row, emb in zip(rows, embeddings, strict=True):
        row["embedding"] = emb

    return db.insert_chunks(rows)
