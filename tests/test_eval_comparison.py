"""Tests for pure eval report comparison logic."""

import pytest

from eval.comparison import CompareError, compare_reports, validate_report


def _result(question: str, rank: int | None) -> dict:
    return {
        "question": question,
        "retrieval": {"first_relevant_rank": rank},
    }


def _report(
    *,
    run_id: str,
    mode: str = "retrieval_only",
    questions_sha256: str = "abc123",
    summary: dict | None = None,
    results: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "questions_sha256": questions_sha256,
        "config": {
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "chat_model": "llama3.2",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "llm_temperature": 0.1,
            **(config or {}),
        },
        "summary": {
            "question_count": 2,
            "hit_rate_at_k": 1.0,
            "precision_at_1": 0.5,
            "mrr": 0.75,
            "ndcg_at_k": 0.8,
            "citation_source_hit_rate": None,
            "citation_precision": None,
            "answer_quality": None,
            **(summary or {}),
        },
        "results": results
        or [
            _result("What is A?", 2),
            _result("What is B?", 1),
        ],
    }


def test_validate_report_rejects_unsupported_schema():
    report = _report(run_id="b")
    report["schema_version"] = 99
    with pytest.raises(CompareError, match="unsupported"):
        validate_report(report, label="baseline")


def test_compare_rejects_different_question_sets():
    baseline = _report(run_id="b", questions_sha256="aaa")
    candidate = _report(run_id="c", questions_sha256="bbb")
    with pytest.raises(CompareError, match="question sets differ"):
        compare_reports(baseline, candidate)


def test_compare_rejects_different_modes_by_default():
    baseline = _report(run_id="b", mode="retrieval_only")
    candidate = _report(run_id="c", mode="full")
    with pytest.raises(CompareError, match="eval modes differ"):
        compare_reports(baseline, candidate)


def test_compare_summary_metric_deltas_and_skips_nulls():
    baseline = _report(
        run_id="b",
        summary={"precision_at_1": 0.5, "mrr": 0.75, "answer_quality": None},
    )
    candidate = _report(
        run_id="c",
        summary={"precision_at_1": 0.8, "mrr": 0.9, "answer_quality": None},
        config={"top_k": 10},
    )

    comparison = compare_reports(baseline, candidate)
    by_name = {metric.name: metric for metric in comparison.metrics}

    assert by_name["precision_at_1"].delta == pytest.approx(0.3)
    assert by_name["mrr"].delta == pytest.approx(0.15)
    assert "answer_quality" not in by_name
    assert len(comparison.config_changes) == 1
    assert comparison.config_changes[0].key == "top_k"
    assert comparison.config_changes[0].baseline == 5
    assert comparison.config_changes[0].candidate == 10


def test_compare_question_rank_improvements_and_regressions():
    baseline = _report(
        run_id="b",
        results=[
            _result("Improved question", 3),
            _result("Regressed question", 1),
            _result("Unchanged question", 2),
            _result("Newly found", None),
            _result("Lost hit", 1),
        ],
    )
    candidate = _report(
        run_id="c",
        results=[
            _result("Improved question", 1),
            _result("Regressed question", 2),
            _result("Unchanged question", 2),
            _result("Newly found", 2),
            _result("Lost hit", None),
        ],
    )

    comparison = compare_reports(baseline, candidate)
    improved = {item.question: item for item in comparison.improved}
    regressed = {item.question: item for item in comparison.regressed}
    unchanged = {item.question for item in comparison.unchanged}

    assert improved["Improved question"].delta == 2
    assert improved["Newly found"].baseline_rank is None
    assert improved["Newly found"].candidate_rank == 2
    assert regressed["Regressed question"].delta == -1
    assert regressed["Lost hit"].candidate_rank is None
    assert "Unchanged question" in unchanged


def test_compare_marks_missing_questions_incomparable():
    baseline = _report(run_id="b", results=[_result("Only baseline", 1)])
    candidate = _report(run_id="c", results=[_result("Only candidate", 1)])

    comparison = compare_reports(baseline, candidate)
    questions = {item.question for item in comparison.incomparable}
    assert questions == {"Only baseline", "Only candidate"}
