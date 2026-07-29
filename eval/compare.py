"""Print the difference between two saved eval reports.

Driven by `make eval-compare BASE=... CANDIDATE=...`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.comparison import CompareError, ComparisonReport, compare_reports


def load_report(path: Path, *, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CompareError(f"cannot read {label} report {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CompareError(f"{label} report {path} is not valid JSON: {exc}") from exc


def _shorten(question: str, width: int = 52) -> str:
    return question if len(question) <= width else question[: width - 3] + "..."


def _rank_label(rank: int | None) -> str:
    return "miss" if rank is None else str(rank)


def format_comparison(comparison: ComparisonReport) -> str:
    lines = [
        f"Baseline:  {comparison.baseline_run_id} ({comparison.mode})",
        f"Candidate: {comparison.candidate_run_id} ({comparison.mode})",
        "",
    ]

    lines.append("Config changes")
    if comparison.config_changes:
        for change in comparison.config_changes:
            lines.append(f"  {change.key}: {change.baseline} -> {change.candidate}")
    else:
        lines.append("  none")
    lines.append("")

    lines.append(f"{'metric':<26}{'baseline':>10}{'candidate':>11}{'delta':>10}")
    for metric in comparison.metrics:
        lines.append(
            f"{metric.name:<26}{metric.baseline:>10.3f}"
            f"{metric.candidate:>11.3f}{metric.delta:>+10.3f}"
        )
    lines.append("")

    for title, changes in (
        ("Improved", comparison.improved),
        ("Regressed", comparison.regressed),
    ):
        lines.append(f"{title} ({len(changes)})")
        if changes:
            for change in changes:
                lines.append(
                    f"  rank {_rank_label(change.baseline_rank)}"
                    f" -> {_rank_label(change.candidate_rank)}"
                    f"  {_shorten(change.question)}"
                )
        else:
            lines.append("  none")
        lines.append("")

    lines.append(f"Unchanged: {len(comparison.unchanged)}")
    if comparison.incomparable:
        lines.append(f"Not in both runs: {len(comparison.incomparable)}")
        for change in comparison.incomparable:
            lines.append(f"  {_shorten(change.question)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline eval report JSON")
    parser.add_argument("candidate", type=Path, help="Candidate eval report JSON")
    parser.add_argument(
        "--allow-mode-mismatch",
        action="store_true",
        help="Compare a retrieval-only run against a full run",
    )
    args = parser.parse_args(argv)

    try:
        comparison = compare_reports(
            load_report(args.baseline, label="baseline"),
            load_report(args.candidate, label="candidate"),
            require_same_mode=not args.allow_mode_mismatch,
        )
    except CompareError as exc:
        print(f"compare failed: {exc}", file=sys.stderr)
        return 1

    print(format_comparison(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
