"""Tests for deterministic source-level retrieval metrics."""

import pytest

from eval.metrics import evaluate_source_ranking, source_matches, unique_sources


def test_source_matches_basename_or_full_path():
    assert source_matches("guide.md", "data/corpus/guide.md")
    assert source_matches("data/corpus/guide.md", "data/corpus/guide.md")
    assert not source_matches("other.md", "data/corpus/guide.md")


def test_unique_sources_keeps_first_occurrence():
    assert unique_sources(["a.md", "a.md", "b.md", "a.md"]) == ["a.md", "b.md"]


def test_perfect_source_ranking():
    metrics = evaluate_source_ranking(["relevant.md", "other.md"], ["relevant.md"])
    assert metrics.hit_at_k is True
    assert metrics.precision_at_1 == 1.0
    assert metrics.reciprocal_rank == 1.0
    assert metrics.ndcg_at_k == 1.0
    assert metrics.first_relevant_rank == 1


def test_relevant_source_lower_in_ranking():
    metrics = evaluate_source_ranking(["other.md", "relevant.md"], ["relevant.md"])
    assert metrics.hit_at_k is True
    assert metrics.precision_at_1 == 0.0
    assert metrics.reciprocal_rank == 0.5
    assert metrics.ndcg_at_k == pytest.approx(1 / 1.5849625007)
    assert metrics.first_relevant_rank == 2


def test_duplicate_chunks_do_not_penalize_source_rank():
    metrics = evaluate_source_ranking(
        ["other.md", "other.md", "relevant.md"],
        ["relevant.md"],
    )
    assert metrics.first_relevant_rank == 2
    assert metrics.reciprocal_rank == 0.5


def test_missed_source_scores_zero():
    metrics = evaluate_source_ranking(["other.md"], ["relevant.md"])
    assert metrics.hit_at_k is False
    assert metrics.precision_at_1 == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.ndcg_at_k == 0.0
    assert metrics.first_relevant_rank is None
