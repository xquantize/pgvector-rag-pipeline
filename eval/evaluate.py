"""Run the pipeline over test_questions.json and score it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.generation import judge_answer
from rag.pipeline import answer_question
from rag.retrieval import retrieve

QUESTIONS_PATH = Path(__file__).resolve().parent / "test_questions.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def retrieval_hit(question: str, relevant_sources: list[str], k: int | None = None) -> bool:
    chunks = retrieve(question, k=k)
    retrieved = {c.source for c in chunks}
    # Match on basename or full path substring.
    for expected in relevant_sources:
        for src in retrieved:
            if expected == src or expected in src or Path(src).name == expected:
                return True
    return False


def run_eval(path: Path = QUESTIONS_PATH) -> int:
    questions = load_questions(path)
    if not questions:
        print("No questions in", path)
        return 1

    hits = 0
    quality_scores: list[float] = []

    print(f"{'#'}: {'question':<50} {'hit':>4} {'quality':>7}")
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

        q_short = (q[:47] + "...") if len(q) > 50 else q
        print(f"{i}: {q_short:<50} {hit!s:>4} {score:>7.2f}")
        print(f"   answer: {result.answer[:200]}")
        print(f"   sources: {', '.join(result.sources) or '(none)'}")

    n = len(questions)
    hit_rate = hits / n
    avg_quality = sum(quality_scores) / n
    print("-" * 70)
    print(f"Retrieval hit rate: {hit_rate:.2%} ({hits}/{n})")
    print(f"Answer quality (avg): {avg_quality:.2f}")
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
