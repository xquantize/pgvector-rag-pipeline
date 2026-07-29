"""Deterministic, source-level retrieval metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path


def source_matches(expected: str, actual: str) -> bool:
    """Match a source label against either a full path or basename."""
    expected_path = Path(expected)
    actual_path = Path(actual)
    if expected_path == actual_path:
        return True
    return expected_path.name == expected and expected == actual_path.name


def unique_sources(sources: list[str]) -> list[str]:
    """Preserve retrieval order while collapsing repeated chunks from one source."""
    return list(dict.fromkeys(sources))


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_k: bool
    precision_at_1: float
    reciprocal_rank: float
    ndcg_at_k: float
    first_relevant_rank: int | None

    def as_dict(self) -> dict[str, bool | float | int | None]:
        return asdict(self)


def evaluate_source_ranking(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> RetrievalMetrics:
    """Score an ordered source ranking against source-level relevance labels."""
    ranked = unique_sources(retrieved_sources)
    relevance = [
        int(any(source_matches(expected, source) for expected in relevant_sources))
        for source in ranked
    ]

    first_rank = next((rank for rank, rel in enumerate(relevance, start=1) if rel), None)
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
    precision_at_1 = float(bool(relevance and relevance[0]))

    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, start=1))
    ideal_relevant = min(len(relevant_sources), len(ranked))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return RetrievalMetrics(
        hit_at_k=first_rank is not None,
        precision_at_1=precision_at_1,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
        first_relevant_rank=first_rank,
    )
