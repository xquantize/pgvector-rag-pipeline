"""Ingestion pipeline: documents -> chunks -> embeddings -> pgvector."""

from __future__ import annotations

import hashlib
import json
import re
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


SKIP_NAMES = {"manifest.json", "readme.md"}


def iter_documents(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and p.name.lower() not in SKIP_NAMES
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_manifest(root: Path) -> dict[str, dict]:
    """Map filename -> {title, category, url} from data/corpus/manifest.json if present."""
    candidates = [
        root / "manifest.json",
        root / "corpus" / "manifest.json",
        root.parent / "corpus" / "manifest.json" if root.name != "corpus" else None,
    ]
    for path in candidates:
        if path and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {row["filename"]: row for row in data}
    return {}


def _meta_from_text(text: str) -> dict[str, str]:
    """Pull title/category/url from HTML-comment headers written by fetch_corpus."""
    meta: dict[str, str] = {}
    for key in ("source", "category"):
        m = re.search(rf"<!--\s*{key}:\s*(.*?)\s*-->", text)
        if m:
            meta["url" if key == "source" else key] = m.group(1).strip()
    title_line = next((ln.strip() for ln in text.splitlines() if ln.startswith("# ")), "")
    if title_line:
        meta["title"] = title_line[2:].strip()
    return meta


def build_metadata(path: Path, text: str, file_hash: str, manifest: dict[str, dict]) -> dict:
    meta = {
        "filename": path.name,
        "file_hash": file_hash,
        **_meta_from_text(text),
    }
    if path.name in manifest:
        entry = manifest[path.name]
        meta.setdefault("title", entry.get("title", path.stem))
        meta.setdefault("category", entry.get("category", "general"))
        meta.setdefault("url", entry.get("url", ""))
    meta.setdefault("title", path.stem)
    meta.setdefault("category", "general")
    return meta


def ingest_path(path: str | Path, *, clear: bool = False) -> int:
    """Load files under `path`, chunk, embed, and upsert. Returns chunk count written."""
    root = Path(path)
    files = iter_documents(root)
    if not files:
        raise FileNotFoundError(f"No supported documents found under {root}")

    if clear:
        db.clear_documents()

    manifest = _load_manifest(root if root.is_dir() else root.parent)
    if root.is_dir() and not clear:
        current_sources = {file_path.as_posix() for file_path in files}
        for source in db.list_sources():
            if Path(source).is_relative_to(root) and source not in current_sources:
                db.delete_source(source)

    total = 0

    for file_path in tqdm(files, desc="ingest"):
        text = load_file(file_path)
        file_hash = _sha256(text)
        source = str(file_path.as_posix())

        existing = db.source_file_hash(source)
        if existing == file_hash:
            tqdm.write(f"skip unchanged: {source}")
            continue

        chunks = chunk_text(text)
        base_meta = build_metadata(file_path, text, file_hash, manifest)

        if not chunks:
            db.delete_source(source)
            continue

        rows = [
            {
                "source": source,
                "chunk_index": idx,
                "content": content,
                "content_hash": _sha256(content),
                "metadata": {**base_meta, "chunk_index": idx},
            }
            for idx, content in enumerate(chunks)
        ]

        embeddings = embed_texts([r["content"] for r in rows])
        for row, emb in zip(rows, embeddings, strict=True):
            row["embedding"] = emb

        total += db.upsert_chunks(rows)
        db.delete_stale_chunks(source, len(chunks))

    return total
