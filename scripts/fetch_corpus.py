"""Fetch a curated Postgres + pgvector docs slice into data/corpus/.

Sources are official PostgreSQL HTML docs and the pgvector README.
Re-run anytime:  make fetch-corpus
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "corpus"

# Curated pages: stable filenames become retrieval `source` basenames in eval.
CORPUS: list[dict[str, str]] = [
    {
        "filename": "pgvector_readme.md",
        "url": "https://raw.githubusercontent.com/pgvector/pgvector/v0.8.0/README.md",
        "title": "pgvector README",
        "category": "pgvector",
        "kind": "markdown",
    },
    {
        "filename": "postgres_indexes_intro.md",
        "url": "https://www.postgresql.org/docs/16/indexes-intro.html",
        "title": "PostgreSQL Index Introduction",
        "category": "indexes",
        "kind": "html",
    },
    {
        "filename": "postgres_indexes_types.md",
        "url": "https://www.postgresql.org/docs/16/indexes-types.html",
        "title": "PostgreSQL Index Types",
        "category": "indexes",
        "kind": "html",
    },
    {
        "filename": "postgres_indexes_multicolumn.md",
        "url": "https://www.postgresql.org/docs/16/indexes-multicolumn.html",
        "title": "PostgreSQL Multicolumn Indexes",
        "category": "indexes",
        "kind": "html",
    },
    {
        "filename": "postgres_textsearch_intro.md",
        "url": "https://www.postgresql.org/docs/16/textsearch-intro.html",
        "title": "PostgreSQL Full Text Search Intro",
        "category": "textsearch",
        "kind": "html",
    },
    {
        "filename": "postgres_textsearch_controls.md",
        "url": "https://www.postgresql.org/docs/16/textsearch-controls.html",
        "title": "PostgreSQL Text Search Controls",
        "category": "textsearch",
        "kind": "html",
    },
    {
        "filename": "postgres_textsearch_indexes.md",
        "url": "https://www.postgresql.org/docs/16/textsearch-indexes.html",
        "title": "PostgreSQL Text Search Indexes",
        "category": "textsearch",
        "kind": "html",
    },
    {
        "filename": "postgres_datatype_json.md",
        "url": "https://www.postgresql.org/docs/16/datatype-json.html",
        "title": "PostgreSQL JSON Types",
        "category": "json",
        "kind": "html",
    },
    {
        "filename": "postgres_functions_json.md",
        "url": "https://www.postgresql.org/docs/16/functions-json.html",
        "title": "PostgreSQL JSON Functions",
        "category": "json",
        "kind": "html",
    },
    {
        "filename": "postgres_using_explain.md",
        "url": "https://www.postgresql.org/docs/16/using-explain.html",
        "title": "PostgreSQL Using EXPLAIN",
        "category": "performance",
        "kind": "html",
    },
    {
        "filename": "postgres_populate.md",
        "url": "https://www.postgresql.org/docs/16/populate.html",
        "title": "PostgreSQL Populating a Database",
        "category": "performance",
        "kind": "html",
    },
    {
        "filename": "postgres_ddl_constraints.md",
        "url": "https://www.postgresql.org/docs/16/ddl-constraints.html",
        "title": "PostgreSQL Constraints",
        "category": "ddl",
        "kind": "html",
    },
    {
        "filename": "postgres_queries_table_expressions.md",
        "url": "https://www.postgresql.org/docs/16/queries-table-expressions.html",
        "title": "PostgreSQL Table Expressions",
        "category": "queries",
        "kind": "html",
    },
    {
        "filename": "postgres_sql_select.md",
        "url": "https://www.postgresql.org/docs/16/sql-select.html",
        "title": "PostgreSQL SELECT",
        "category": "queries",
        "kind": "html",
    },
    {
        "filename": "postgres_datatype_numeric.md",
        "url": "https://www.postgresql.org/docs/16/datatype-numeric.html",
        "title": "PostgreSQL Numeric Types",
        "category": "types",
        "kind": "html",
    },
    {
        "filename": "postgres_transaction_iso.md",
        "url": "https://www.postgresql.org/docs/16/transaction-iso.html",
        "title": "PostgreSQL Transaction Isolation",
        "category": "mvcc",
        "kind": "html",
    },
    {
        "filename": "postgres_explicit_locking.md",
        "url": "https://www.postgresql.org/docs/16/explicit-locking.html",
        "title": "PostgreSQL Explicit Locking",
        "category": "mvcc",
        "kind": "html",
    },
]


class _DocTextExtractor(HTMLParser):
    """Pull text only from #docContent."""

    SKIP: ClassVar[set[str]] = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._div_depth = 0
        self._in_doc = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: (v or "") for k, v in attrs}
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "div" and attrs_dict.get("id") == "docContent":
            self._in_doc = True
            self._div_depth = 1
            return
        if self._in_doc and tag == "div":
            self._div_depth += 1
        if not self._in_doc:
            return
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "tr", "br", "pre", "dt", "dd", "div"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth or not self._in_doc:
            return
        if tag == "div":
            self._div_depth -= 1
            if self._div_depth <= 0:
                self._in_doc = False
                self._div_depth = 0

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._in_doc:
            return
        text = data.strip()
        if text:
            self._parts.append(text + " ")

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_markdown(html: str, *, title: str, url: str, category: str) -> str:
    parser = _DocTextExtractor()
    parser.feed(html)
    body = parser.text()
    if len(body) < 400:
        # Slice between docContent and docComments as a fallback.
        m = re.search(
            r'id="docContent"(.*?)id="docComments"',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        chunk = m.group(1) if m else html
        plain = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", chunk)
        plain = re.sub(r"(?s)<[^>]+>", " ", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        body = plain
    return (
        f"# {title}\n\n"
        f"<!-- source: {url} -->\n"
        f"<!-- category: {category} -->\n"
        f"<!-- license: PostgreSQL License / project LICENSE as applicable -->\n\n"
        f"{body}\n"
    )


def wrap_markdown(md: str, *, title: str, url: str, category: str) -> str:
    if md.lstrip().startswith("#"):
        body = md
    else:
        body = f"# {title}\n\n{md}"
    header = (
        f"<!-- source: {url} -->\n"
        f"<!-- category: {category} -->\n"
        f"<!-- license: see upstream project LICENSE -->\n\n"
    )
    if "<!-- source:" in body[:400]:
        return body if body.endswith("\n") else body + "\n"
    return header + body


def fetch_one(client: httpx.Client, entry: dict[str, str]) -> str:
    resp = client.get(entry["url"])
    resp.raise_for_status()
    if entry["kind"] == "markdown":
        return wrap_markdown(
            resp.text,
            title=entry["title"],
            url=entry["url"],
            category=entry["category"],
        )
    return html_to_markdown(
        resp.text,
        title=entry["title"],
        url=entry["url"],
        category=entry["category"],
    )


def fetch_corpus(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop old demo file if present at data root; corpus lives here.
    written = 0
    headers = {"User-Agent": "pgvector-rag-pipeline-corpus-fetcher/0.1 (research; local eval)"}
    with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
        for entry in CORPUS:
            path = out_dir / entry["filename"]
            print(f"fetch {entry['filename']} ...", flush=True)
            text = fetch_one(client, entry)
            path.write_text(text, encoding="utf-8")
            written += 1
            print(f"  -> {path.relative_to(ROOT)} ({len(text)} chars)")
    # Manifest for ingest metadata
    manifest = out_dir / "manifest.json"
    import json

    manifest.write_text(
        json.dumps(
            [
                {
                    "filename": e["filename"],
                    "url": e["url"],
                    "title": e["title"],
                    "category": e["category"],
                }
                for e in CORPUS
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest.relative_to(ROOT)}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)
    try:
        n = fetch_corpus(args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"fetch-corpus failed: {exc}", file=sys.stderr)
        return 1
    print(f"Fetched {n} documents into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
