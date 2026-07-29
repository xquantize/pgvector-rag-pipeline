"""CLI entry point: build the index from ./data.  Usage: make ingest"""

from __future__ import annotations

import argparse
import sys

from rag.ingest import ingest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into pgvector.")
    parser.add_argument(
        "path",
        nargs="?",
        default="./data",
        help="File or directory to ingest (default: ./data)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Truncate the documents table before inserting",
    )
    args = parser.parse_args(argv)

    try:
        n = ingest_path(args.path, clear=args.clear)
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    print(f"Ingested {n} chunks from {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
