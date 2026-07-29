"""Run retrieval/generation evals and save reproducible result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.metrics import evaluate_source_ranking, source_matches, unique_sources
from eval.reporting import write_run_artifacts
from rag.config import settings
from rag.generation import AnswerWithSources, generate_answer, judge_answer
from rag.retrieval import retrieve

QUESTIONS_PATH = Path(__file__).resolve().parent / "test_questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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
    cited_sources = {result.chunks[i - 1].source for i in result.citations if 1 <= i <= n}
    return any(
        source_matches(expected, src) for expected in relevant_sources for src in cited_sources
    )


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _run_config(k: int) -> dict[str, Any]:
    return {
        "retrieval_mode": settings.retrieval_mode,
        "top_k": k,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": (
            settings.ollama_embed_model
            if settings.embedding_provider == "ollama"
            else settings.hf_embed_model
        ),
        "chat_model": settings.ollama_chat_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "llm_temperature": settings.llm_temperature,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str | None:
    """Best-effort code revision for reproducible local/CI reports."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_eval(
    path: Path = QUESTIONS_PATH,
    *,
    k: int | None = None,
    retrieval_only: bool = False,
    output_dir: Path = RESULTS_PATH,
    save_artifacts: bool = True,
) -> int:
    questions = load_questions(path)
    if not questions:
        print("No questions in", path)
        return 1

    top_k = k if k is not None else settings.top_k
    if top_k < 1:
        raise ValueError("k must be at least 1")

    started = time.perf_counter()
    run_time = datetime.now(UTC)
    milliseconds = run_time.microsecond // 1000
    run_id = f"{run_time:%Y%m%dT%H%M%S}{milliseconds:03d}Z"
    results: list[dict[str, Any]] = []

    hit_scores: list[float] = []
    precision_at_1_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    quality_scores: list[float] = []
    cite_hit_scores: list[float] = []
    cite_precision_scores: list[float] = []

    print(
        f"{'#':>2}  {'question':<38} {'rank':>4} {'p@1':>4} "
        f"{'rr':>4} {'ndcg':>5} {'cite':>4} {'qual':>5}"
    )
    print("-" * 84)

    for i, item in enumerate(questions, start=1):
        q = item["question"]
        expected = item.get("expected_answer", "")
        relevant = item.get("relevant_sources") or []

        chunks = retrieve(q, k=top_k)
        retrieved_sources = unique_sources([chunk.source for chunk in chunks])
        retrieval_metrics = evaluate_source_ranking(retrieved_sources, relevant)

        hit_scores.append(float(retrieval_metrics.hit_at_k))
        precision_at_1_scores.append(retrieval_metrics.precision_at_1)
        reciprocal_ranks.append(retrieval_metrics.reciprocal_rank)
        ndcg_scores.append(retrieval_metrics.ndcg_at_k)

        answer: str | None = None
        citations: list[int] = []
        c_prec: float | None = None
        c_hit: bool | None = None
        score: float | None = None

        if not retrieval_only:
            answer_result = generate_answer(q, chunks)
            answer = answer_result.answer
            citations = answer_result.citations
            c_prec = citation_precision(answer_result)
            c_hit = citation_source_hit(answer_result, relevant)
            score = judge_answer(q, expected, answer) if expected else 0.0
            cite_precision_scores.append(c_prec)
            cite_hit_scores.append(float(c_hit))
            quality_scores.append(score)

        result_row = {
            "question": q,
            "expected_answer": expected,
            "relevant_sources": relevant,
            "retrieved_sources": retrieved_sources,
            "retrieval": retrieval_metrics.as_dict(),
            "answer": answer,
            "citations": citations,
            "citation_source_hit": c_hit,
            "citation_precision": c_prec,
            "judge_score": score,
        }
        results.append(result_row)

        q_short = (q[:35] + "...") if len(q) > 38 else q
        rank = retrieval_metrics.first_relevant_rank or "-"
        cite_display = "-" if c_hit is None else str(c_hit)
        quality_display = "-" if score is None else f"{score:.2f}"
        print(
            f"{i:>2}  {q_short:<38} {rank!s:>4} "
            f"{retrieval_metrics.precision_at_1:>4.2f} "
            f"{retrieval_metrics.reciprocal_rank:>4.2f} "
            f"{retrieval_metrics.ndcg_at_k:>5.2f} "
            f"{cite_display:>4} {quality_display:>5}"
        )
        if answer is not None:
            print(f"    answer: {answer[:200]}")
            print(f"    sources: {', '.join(retrieved_sources) or '(none)'}")
            if citations:
                print(f"    citations: {citations} (precision={c_prec:.2f})")

    n = len(questions)
    summary: dict[str, int | float | None] = {
        "question_count": n,
        "hit_rate_at_k": _average(hit_scores),
        "precision_at_1": _average(precision_at_1_scores),
        "mrr": _average(reciprocal_ranks),
        "ndcg_at_k": _average(ndcg_scores),
        "citation_source_hit_rate": (_average(cite_hit_scores) if cite_hit_scores else None),
        "citation_precision": (_average(cite_precision_scores) if cite_precision_scores else None),
        "answer_quality": _average(quality_scores) if quality_scores else None,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": run_time.isoformat(),
        "mode": "retrieval_only" if retrieval_only else "full",
        "questions_path": str(path),
        "questions_sha256": _file_sha256(path),
        "code_revision": _git_revision(),
        "config": _run_config(top_k),
        "summary": summary,
        "results": results,
    }

    print("-" * 84)
    print(f"Hit rate@{top_k}:       {summary['hit_rate_at_k']:.2%}")
    print(f"Precision@1:       {summary['precision_at_1']:.2%}")
    print(f"MRR:               {summary['mrr']:.3f}")
    print(f"nDCG@{top_k}:           {summary['ndcg_at_k']:.3f}")
    if not retrieval_only:
        print(f"Citation source hit: {summary['citation_source_hit_rate']:.2%}")
        print(f"Citation precision:  {summary['citation_precision']:.3f}")
        print(f"Answer quality:      {summary['answer_quality']:.3f}")
        print("Judge quality is noisy; retrieval/citation metrics are deterministic.")

    if save_artifacts:
        json_path, csv_path = write_run_artifacts(report, output_dir)
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV:  {csv_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "questions",
        nargs="?",
        type=Path,
        default=QUESTIONS_PATH,
        help=f"Question set (default: {QUESTIONS_PATH})",
    )
    parser.add_argument("--k", type=int, help="Override retrieval top-k")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and judge calls for a fast retrieval eval",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_PATH,
        help=f"Artifact directory (default: {RESULTS_PATH})",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Print results without writing JSON/CSV artifacts",
    )
    args = parser.parse_args(argv)
    try:
        return run_eval(
            args.questions,
            k=args.k,
            retrieval_only=args.retrieval_only,
            output_dir=args.output_dir,
            save_artifacts=not args.no_artifacts,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"eval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
