"""Tests for eval JSON and CSV artifacts."""

import csv
import json

from eval.reporting import write_run_artifacts


def test_write_run_artifacts(tmp_path):
    report = {
        "run_id": "20260729T010203Z",
        "results": [
            {
                "question": "What is it?",
                "relevant_sources": ["expected.md"],
                "retrieved_sources": ["expected.md", "other.md"],
                "retrieval": {
                    "hit_at_k": True,
                    "precision_at_1": 1.0,
                    "reciprocal_rank": 1.0,
                    "ndcg_at_k": 1.0,
                    "first_relevant_rank": 1,
                },
                "citation_source_hit": True,
                "citation_precision": 1.0,
                "judge_score": 0.8,
                "answer": "A fact [1].",
            }
        ],
    }

    json_path, csv_path = write_run_artifacts(report, tmp_path)

    assert json.loads(json_path.read_text()) == report
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["question"] == "What is it?"
    assert rows[0]["retrieved_sources"] == "expected.md|other.md"
    assert rows[0]["reciprocal_rank"] == "1.0"
