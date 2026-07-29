"""Compare two saved eval reports without re-running the pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

SUPPORTED_SCHEMA_VERSION = 1
NOT_RECORDED = "<not recorded>"

SUMMARY_METRICS = (
    "hit_rate_at_k",
    "precision_at_1",
    "mrr",
    "ndcg_at_k",
    "citation_source_hit_rate",
    "citation_precision",
    "answer_quality",
)


class CompareError(ValueError):
    """Raised when two reports cannot be compared safely."""


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float
    candidate: float
    delta: float


@dataclass(frozen=True)
class QuestionRankChange:
    question: str
    baseline_rank: int | None
    candidate_rank: int | None
    delta: int | None
    status: Literal["improved", "regressed", "unchanged"]


@dataclass(frozen=True)
class ConfigChange:
    key: str
    baseline: Any
    candidate: Any


@dataclass(frozen=True)
class ComparisonReport:
    baseline_run_id: str
    candidate_run_id: str
    baseline_revision: str | None
    candidate_revision: str | None
    questions_sha256: str
    mode: str
    config_changes: list[ConfigChange]
    metrics: list[MetricDelta]
    improved: list[QuestionRankChange]
    regressed: list[QuestionRankChange]
    unchanged: list[QuestionRankChange]


def _require_mapping(report: Any, label: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise CompareError(f"{label} report must be a JSON object")
    return report


def _require_key(report: dict[str, Any], key: str, label: str) -> Any:
    if key not in report:
        raise CompareError(
            f"{label} report is missing required field: {key} "
            "(reports written by an older version need re-running)"
        )
    return report[key]


def validate_report(report: Any, *, label: str) -> dict[str, Any]:
    """Validate the minimum fields needed for a safe comparison."""
    report = _require_mapping(report, label)
    schema_version = _require_key(report, "schema_version", label)
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise CompareError(
            f"{label} report schema_version={schema_version!r} is unsupported "
            f"(expected {SUPPORTED_SCHEMA_VERSION})"
        )

    for key in ("run_id", "mode", "questions_sha256", "config", "summary", "results"):
        _require_key(report, key, label)

    for key in ("run_id", "mode", "questions_sha256"):
        if not isinstance(report[key], str) or not report[key]:
            raise CompareError(f"{label} report {key} must be a non-empty string")
    revision = report.get("code_revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise CompareError(f"{label} report code_revision must be a non-empty string or null")
    if not isinstance(report["config"], dict):
        raise CompareError(f"{label} report config must be an object")
    if not isinstance(report["summary"], dict):
        raise CompareError(f"{label} report summary must be an object")
    if not isinstance(report["results"], list) or not report["results"]:
        raise CompareError(f"{label} report results must be a non-empty list")

    questions: set[str] = set()
    for index, row in enumerate(report["results"]):
        if not isinstance(row, dict):
            raise CompareError(f"{label} result[{index}] must be an object")
        question = row.get("question")
        if not isinstance(question, str) or not question:
            raise CompareError(f"{label} result[{index}].question must be a non-empty string")
        if question in questions:
            raise CompareError(f"{label} report contains duplicate question: {question!r}")
        questions.add(question)
        retrieval = row.get("retrieval")
        if not isinstance(retrieval, dict):
            raise CompareError(f"{label} result[{index}] is missing retrieval metrics")
        if "first_relevant_rank" not in retrieval:
            raise CompareError(f"{label} result[{index}] is missing retrieval.first_relevant_rank")
        rank = retrieval["first_relevant_rank"]
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
            raise CompareError(
                f"{label} result[{index}].retrieval.first_relevant_rank "
                "must be a positive integer or null"
            )

    return report


def _metric_deltas(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> list[MetricDelta]:
    deltas: list[MetricDelta] = []
    for name in SUMMARY_METRICS:
        baseline_value = baseline_summary.get(name)
        candidate_value = candidate_summary.get(name)
        if baseline_value is None or candidate_value is None:
            continue
        if isinstance(baseline_value, bool) or isinstance(candidate_value, bool):
            raise CompareError(f"summary.{name} must be numeric when present")
        try:
            baseline_num = float(baseline_value)
            candidate_num = float(candidate_value)
        except (TypeError, ValueError) as exc:
            raise CompareError(f"summary.{name} must be numeric when present") from exc
        if not math.isfinite(baseline_num) or not math.isfinite(candidate_num):
            raise CompareError(f"summary.{name} must be finite when present")
        if not 0.0 <= baseline_num <= 1.0 or not 0.0 <= candidate_num <= 1.0:
            raise CompareError(f"summary.{name} must be between 0 and 1")
        deltas.append(
            MetricDelta(
                name=name,
                baseline=baseline_num,
                candidate=candidate_num,
                delta=candidate_num - baseline_num,
            )
        )
    return deltas


def _config_changes(
    baseline_config: dict[str, Any],
    candidate_config: dict[str, Any],
) -> list[ConfigChange]:
    changes: list[ConfigChange] = []
    for key in sorted(baseline_config.keys() | candidate_config.keys()):
        baseline_value = baseline_config.get(key, NOT_RECORDED)
        candidate_value = candidate_config.get(key, NOT_RECORDED)
        if baseline_value != candidate_value:
            changes.append(
                ConfigChange(key=key, baseline=baseline_value, candidate=candidate_value)
            )
    return changes


def _rank_status(
    baseline_rank: int | None, candidate_rank: int | None
) -> Literal["improved", "regressed", "unchanged"]:
    if baseline_rank is None and candidate_rank is None:
        return "unchanged"
    if baseline_rank is None and candidate_rank is not None:
        return "improved"
    if baseline_rank is not None and candidate_rank is None:
        return "regressed"
    assert baseline_rank is not None and candidate_rank is not None
    if candidate_rank < baseline_rank:
        return "improved"
    if candidate_rank > baseline_rank:
        return "regressed"
    return "unchanged"


def _question_rank_changes(
    baseline_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
) -> tuple[
    list[QuestionRankChange],
    list[QuestionRankChange],
    list[QuestionRankChange],
]:
    baseline_by_question = {row["question"]: row for row in baseline_results}
    candidate_by_question = {row["question"]: row for row in candidate_results}

    if baseline_by_question.keys() != candidate_by_question.keys():
        raise CompareError("report results contain different questions")

    improved: list[QuestionRankChange] = []
    regressed: list[QuestionRankChange] = []
    unchanged: list[QuestionRankChange] = []

    for question in baseline_by_question:
        baseline_rank = baseline_by_question[question]["retrieval"].get("first_relevant_rank")
        candidate_rank = candidate_by_question[question]["retrieval"].get("first_relevant_rank")

        if baseline_rank is None or candidate_rank is None:
            delta = None
        else:
            # Lower rank is better, so positive delta means improvement.
            delta = baseline_rank - candidate_rank

        change = QuestionRankChange(
            question=question,
            baseline_rank=baseline_rank,
            candidate_rank=candidate_rank,
            delta=delta,
            status=_rank_status(baseline_rank, candidate_rank),
        )
        if change.status == "improved":
            improved.append(change)
        elif change.status == "regressed":
            regressed.append(change)
        else:
            unchanged.append(change)

    return improved, regressed, unchanged


def compare_reports(
    baseline: Any,
    candidate: Any,
) -> ComparisonReport:
    """Compare two validated eval reports and return structured deltas."""
    baseline = validate_report(baseline, label="baseline")
    candidate = validate_report(candidate, label="candidate")

    if baseline["questions_sha256"] != candidate["questions_sha256"]:
        raise CompareError(
            "question sets differ "
            f"(baseline={baseline['questions_sha256'][:12]}… "
            f"candidate={candidate['questions_sha256'][:12]}…)"
        )

    if baseline["mode"] != candidate["mode"]:
        raise CompareError(
            f"eval modes differ (baseline={baseline['mode']!r}, candidate={candidate['mode']!r})"
        )

    improved, regressed, unchanged = _question_rank_changes(
        baseline["results"],
        candidate["results"],
    )

    return ComparisonReport(
        baseline_run_id=str(baseline["run_id"]),
        candidate_run_id=str(candidate["run_id"]),
        baseline_revision=baseline.get("code_revision"),
        candidate_revision=candidate.get("code_revision"),
        questions_sha256=str(baseline["questions_sha256"]),
        mode=str(candidate["mode"]),
        config_changes=_config_changes(baseline["config"], candidate["config"]),
        metrics=_metric_deltas(baseline["summary"], candidate["summary"]),
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
    )
