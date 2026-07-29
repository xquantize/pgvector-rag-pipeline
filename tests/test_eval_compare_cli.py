"""Tests for the eval comparison entry point used by `make eval-compare`."""

import json

import pytest

from eval.compare import format_comparison, latest_report_paths, load_report, main
from eval.comparison import CompareError, compare_reports


def _report(run_id: str, ranks: dict[str, int | None], **summary) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "code_revision": f"{run_id}-revision",
        "mode": "retrieval_only",
        "questions_sha256": "abc123",
        "config": {"top_k": 5, "retrieval_mode": "hybrid"},
        "summary": {"precision_at_1": 0.5, "mrr": 0.75, **summary},
        "results": [
            {"question": question, "retrieval": {"first_relevant_rank": rank}}
            for question, rank in ranks.items()
        ],
    }


def _write(tmp_path, name: str, report: dict):
    path = tmp_path / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_load_report_missing_file(tmp_path):
    with pytest.raises(CompareError, match="cannot read baseline report"):
        load_report(tmp_path / "nope.json", label="baseline")


def test_load_report_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CompareError, match="not valid JSON"):
        load_report(path, label="candidate")


def test_latest_report_paths_uses_timestamped_names(tmp_path):
    oldest = _write(tmp_path, "eval-20260101T000000000Z.json", {})
    previous = _write(tmp_path, "eval-20260102T000000000Z.json", {})
    latest = _write(tmp_path, "eval-20260103T000000000Z.json", {})
    _write(tmp_path, "unrelated.json", {})

    assert latest_report_paths(tmp_path) == (previous, latest)
    assert oldest not in latest_report_paths(tmp_path)


def test_latest_report_paths_requires_two_reports(tmp_path):
    _write(tmp_path, "eval-20260101T000000000Z.json", {})
    with pytest.raises(CompareError, match="need two reports"):
        latest_report_paths(tmp_path)


def test_format_comparison_reports_changes():
    baseline = _report("base", {"Q improved": 3, "Q same": 1}, precision_at_1=0.5)
    baseline["code_revision"] = f"{'a' * 40}-dirty"
    comparison = compare_reports(
        baseline,
        _report("cand", {"Q improved": 1, "Q same": 1}, precision_at_1=1.0),
    )
    output = format_comparison(comparison)

    assert "Baseline:  base" in output
    assert "Candidate: cand" in output
    assert "aaaaaaaaaaaa-dirty" in output
    assert "precision_at_1" in output
    assert "+0.500" in output
    assert "Improved (1)" in output
    assert "rank 3 -> 1" in output
    assert "Regressed (0)" in output
    assert "Unchanged: 1" in output


def test_format_comparison_marks_lost_hit_as_miss():
    comparison = compare_reports(
        _report("base", {"Q": 2}),
        _report("cand", {"Q": None}),
    )
    output = format_comparison(comparison)
    assert "Regressed (1)" in output
    assert "rank 2 -> miss" in output


def test_main_returns_zero_for_comparable_runs(tmp_path, capsys):
    baseline = _write(tmp_path, "base.json", _report("base", {"Q": 2}))
    candidate = _write(tmp_path, "cand.json", _report("cand", {"Q": 1}))

    exit_code = main([str(baseline), str(candidate)])

    assert exit_code == 0
    assert "Improved (1)" in capsys.readouterr().out


def test_main_latest_compares_newest_two_runs(tmp_path, capsys):
    _write(tmp_path, "eval-20260101T000000000Z.json", _report("old", {"Q": 3}))
    _write(tmp_path, "eval-20260102T000000000Z.json", _report("base", {"Q": 2}))
    _write(tmp_path, "eval-20260103T000000000Z.json", _report("cand", {"Q": 1}))

    exit_code = main(["--latest", "--results-dir", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Baseline:  base" in output
    assert "Candidate: cand" in output


def test_main_returns_one_on_mismatched_question_sets(tmp_path, capsys):
    baseline = _write(tmp_path, "base.json", _report("base", {"Q": 1}))
    candidate_report = _report("cand", {"Q": 1})
    candidate_report["questions_sha256"] = "different"
    candidate = _write(tmp_path, "cand.json", candidate_report)

    exit_code = main([str(baseline), str(candidate)])

    assert exit_code == 1
    assert "question sets differ" in capsys.readouterr().err
