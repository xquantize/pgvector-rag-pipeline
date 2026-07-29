"""Compare two saved eval reports without re-running the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_SCHEMA_VERSION = 1

SUMMARY_METRICS = (
    "hit_rate_at_k",
    "precision_at_1",
    "mrr",
    "ndcg_at_k",
    "citation_source_hit_rate",
    "citation_precision",
    "answer_quality",
)

CONFIG_KEYS = (
    "retrieval_mode",
    "top_k",
    "embedding_provider",
    "embedding_model",
    "chat_model",
    "chunk_size",
    "chunk_overlap",
    "llm_temperature",
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
    status: str  # improved | regressed | unchanged | incomparable


@dataclass(frozen=True)
class ConfigChange:
    key: str
    baseline: Any
    candidate: Any


@dataclass(frozen=True)
class ComparisonReport:
    baseline_run_id: str
    candidate_run_id: str
    questions_sha256: str
    mode: str
    config_changes: list[ConfigChange]
    metrics: list[MetricDelta]
    improved: list[QuestionRankChange]
    regressed: list[QuestionRankChange]
    unchanged: list[QuestionRankChange]
    incomparable: list[QuestionRankChange]


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


def validate_report(report: dict[str, Any], *, label: str) -> dict[str, Any]:
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

    if not isinstance(report["config"], dict):
        raise CompareError(f"{label} report config must be an object")
    if not isinstance(report["summary"], dict):
        raise CompareError(f"{label} report summary must be an object")
    if not isinstance(report["results"], list) or not report["results"]:
        raise CompareError(f"{label} report results must be a non-empty list")

    for index, row in enumerate(report["results"]):
        if not isinstance(row, dict):
            raise CompareError(f"{label} result[{index}] must be an object")
        if "question" not in row:
            raise CompareError(f"{label} result[{index}] is missing question")
        retrieval = row.get("retrieval")
        if not isinstance(retrieval, dict):
            raise CompareError(f"{label} result[{index}] is missing retrieval metrics")
        if "first_relevant_rank" not in retrieval:
            raise CompareError(f"{label} result[{index}] is missing retrieval.first_relevant_rank")

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
        try:
            baseline_num = float(baseline_value)
            candidate_num = float(candidate_value)
        except (TypeError, ValueError) as exc:
            raise CompareError(f"summary.{name} must be numeric when present") from exc
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
    for key in CONFIG_KEYS:
        baseline_value = baseline_config.get(key)
        candidate_value = candidate_config.get(key)
        if baseline_value != candidate_value:
            changes.append(
                ConfigChange(key=key, baseline=baseline_value, candidate=candidate_value)
            )
    return changes


def _rank_status(baseline_rank: int | None, candidate_rank: int | None) -> str:
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
    list[QuestionRankChange],
]:
    baseline_by_question = {row["question"]: row for row in baseline_results}
    candidate_by_question = {row["question"]: row for row in candidate_results}

    shared = [q for q in baseline_by_question if q in candidate_by_question]
    only_baseline = [q for q in baseline_by_question if q not in candidate_by_question]
    only_candidate = [q for q in candidate_by_question if q not in baseline_by_question]

    improved: list[QuestionRankChange] = []
    regressed: list[QuestionRankChange] = []
    unchanged: list[QuestionRankChange] = []
    incomparable: list[QuestionRankChange] = []

    for question in only_baseline + only_candidate:
        incomparable.append(
            QuestionRankChange(
                question=question,
                baseline_rank=None,
                candidate_rank=None,
                delta=None,
                status="incomparable",
            )
        )

    for question in shared:
        baseline_rank = baseline_by_question[question]["retrieval"].get("first_relevant_rank")
        candidate_rank = candidate_by_question[question]["retrieval"].get("first_relevant_rank")
        if baseline_rank is not None:
            baseline_rank = int(baseline_rank)
        if candidate_rank is not None:
            candidate_rank = int(candidate_rank)

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

    return improved, regressed, unchanged, incomparable


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    require_same_mode: bool = True,
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

    if require_same_mode and baseline["mode"] != candidate["mode"]:
        raise CompareError(
            f"eval modes differ (baseline={baseline['mode']!r}, "
            f"candidate={candidate['mode']!r}); pass require_same_mode=False to allow"
        )

    improved, regressed, unchanged, incomparable = _question_rank_changes(
        baseline["results"],
        candidate["results"],
    )

    return ComparisonReport(
        baseline_run_id=str(baseline["run_id"]),
        candidate_run_id=str(candidate["run_id"]),
        questions_sha256=str(baseline["questions_sha256"]),
        mode=str(candidate["mode"]),
        config_changes=_config_changes(baseline["config"], candidate["config"]),
        metrics=_metric_deltas(baseline["summary"], candidate["summary"]),
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
        incomparable=incomparable,
    )
