"""Run the pipeline over test_questions.json and score it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.generation import AnswerWithSources, judge_answer
from rag.pipeline import answer_question
from rag.retrieval import retrieve

QUESTIONS_PATH = Path(__file__).resolve().parent / "test_questions.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _source_match(expected: str, source: str) -> bool:
    return expected == source or expected in source or Path(source).name == expected


def retrieval_hit(question: str, relevant_sources: list[str], k: int | None = None) -> bool:
    chunks = retrieve(question, k=k)
    retrieved = {c.source for c in chunks}
    return any(
        _source_match(expected, src) for expected in relevant_sources for src in retrieved
    )


def citation_precision(result: AnswerWithSources) -> float:
    """Share of [n] markers that point at a real retrieved chunk."""
    if not result.citations:
        return 0.0
    n = len(result.chunks)
    valid = sum(1 for i in result.citations if 1 <= i <= n)
    return valid / len(result.citations)


def citation_source_hit(result: AnswerWithSources, relevant_sources: list[str]) -> bool:
    """True if any cited chunk comes from an expected source file."""
    if not result.citations or not result.chunks:
        return False
    n = len(result.chunks)
    cited_sources = {
        result.chunks[i - 1].source for i in result.citations if 1 <= i <= n
    }
    return any(
        _source_match(expected, src)
        for expected in relevant_sources
        for src in cited_sources
    )


def run_eval(path: Path = QUESTIONS_PATH) -> int:
    questions = load_questions(path)
    if not questions:
        print("No questions in", path)
        return 1

    hits = 0
    cite_hits = 0
    quality_scores: list[float] = []
    cite_precision_scores: list[float] = []

    print(f"{'#':>2}  {'question':<44} {'hit':>4} {'cite':>4} {'qual':>5}")
    print("-" * 70)

    for i, item in enumerate(questions, start=1):
        q = item["question"]
        expected = item.get("expected_answer", "")
        relevant = item.get("relevant_sources") or []

        hit = retrieval_hit(q, relevant)
        hits += int(hit)

        result = answer_question(q)
        score = judge_answer(q, expected, result.answer) if expected else 0.0
        quality_scores.append(score)

        c_prec = citation_precision(result)
        c_hit = citation_source_hit(result, relevant)
        cite_precision_scores.append(c_prec)
        cite_hits += int(c_hit)

        q_short = (q[:41] + "...") if len(q) > 44 else q
        print(f"{i:>2}  {q_short:<44} {hit!s:>4} {c_hit!s:>4} {score:>5.2f}")
        print(f"    answer: {result.answer[:200]}")
        print(f"    sources: {', '.join(result.sources) or '(none)'}")
        if result.citations:
            print(f"    citations: {result.citations} (precision={c_prec:.2f})")

    n = len(questions)
    print("-" * 70)
    print(f"Retrieval hit rate:     {hits / n:.2%} ({hits}/{n})")
    print(f"Citation source hit:    {cite_hits / n:.2%} ({cite_hits}/{n})")
    print(f"Citation precision avg: {sum(cite_precision_scores) / n:.2f}")
    print(f"Answer quality (judge): {sum(quality_scores) / n:.2f}")
    print()
    print(
        "Note: judge score is noisy (same local model grading itself). "
        "Citation metrics are deterministic."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    path = Path(args[0]) if args else QUESTIONS_PATH
    try:
        return run_eval(path)
    except Exception as exc:  # noqa: BLE001
        print(f"eval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
