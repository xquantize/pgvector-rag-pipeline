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

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_report(path: Path, *, label: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CompareError(f"cannot read {label} report {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CompareError(f"{label} report {path} is not valid JSON: {exc}") from exc


def latest_report_paths(directory: Path) -> tuple[Path, Path]:
    """Return the previous and latest report using timestamped filenames."""
    reports = sorted(directory.glob("eval-*.json"))
    if len(reports) < 2:
        raise CompareError(
            f"need two reports in {directory} (found {len(reports)}); run 'make eval-retrieval'"
        )
    return reports[-2], reports[-1]


def _shorten(question: str, width: int = 52) -> str:
    return question if len(question) <= width else question[: width - 3] + "..."


def _rank_label(rank: int | None) -> str:
    return "miss" if rank is None else str(rank)


def _revision_label(revision: str | None) -> str:
    if not revision:
        return ""
    dirty = revision.endswith("-dirty")
    clean = revision.removesuffix("-dirty")
    short = clean if len(clean) <= 18 else clean[:12]
    return f", code {short}{'-dirty' if dirty else ''}"


def format_comparison(comparison: ComparisonReport) -> str:
    lines = [
        (
            f"Baseline:  {comparison.baseline_run_id} "
            f"({comparison.mode}{_revision_label(comparison.baseline_revision)})"
        ),
        (
            f"Candidate: {comparison.candidate_run_id} "
            f"({comparison.mode}{_revision_label(comparison.candidate_revision)})"
        ),
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

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path, help="Baseline eval report JSON")
    parser.add_argument("candidate", nargs="?", type=Path, help="Candidate eval report JSON")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Compare the two latest reports",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    try:
        if args.latest:
            if args.baseline is not None or args.candidate is not None:
                raise CompareError("--latest cannot be combined with report paths")
            baseline_path, candidate_path = latest_report_paths(args.results_dir)
        elif args.baseline is None or args.candidate is None:
            raise CompareError(
                "provide BASE and CANDIDATE reports, or use 'make eval-compare-latest'"
            )
        else:
            baseline_path, candidate_path = args.baseline, args.candidate

        comparison = compare_reports(
            load_report(baseline_path, label="baseline"),
            load_report(candidate_path, label="candidate"),
        )
    except CompareError as exc:
        print(f"compare failed: {exc}", file=sys.stderr)
        return 1

    print(format_comparison(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
