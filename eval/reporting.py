"""Write reproducible eval run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "question",
    "relevant_sources",
    "retrieved_sources",
    "hit_at_k",
    "precision_at_1",
    "reciprocal_rank",
    "ndcg_at_k",
    "first_relevant_rank",
    "citation_source_hit",
    "citation_precision",
    "judge_score",
    "answer",
]


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    retrieval = result["retrieval"]
    return {
        "question": result["question"],
        "relevant_sources": "|".join(result["relevant_sources"]),
        "retrieved_sources": "|".join(result["retrieved_sources"]),
        **retrieval,
        "citation_source_hit": result.get("citation_source_hit"),
        "citation_precision": result.get("citation_precision"),
        "judge_score": result.get("judge_score"),
        "answer": result.get("answer"),
    }


def write_run_artifacts(
    report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write a full JSON report and a flat per-question CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval-{report['run_id']}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(result) for result in report["results"])

    return json_path, csv_path
