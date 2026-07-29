"""CLI entry point: ask the corpus a question.  Usage: make ask Q="..." """

from __future__ import annotations

import argparse
import sys

from rag.pipeline import answer_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer a question from the ingested corpus.")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("-k", type=int, default=None, help="Chunks to retrieve (default: top_k)")
    args = parser.parse_args(argv)

    try:
        result = answer_question(args.question, k=args.k)
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"query failed: {exc}", file=sys.stderr)
        return 1

    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"  {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
