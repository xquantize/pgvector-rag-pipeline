"""Unit tests for eval citation helpers."""

from eval.evaluate import citation_precision, citation_source_hit
from rag.generation import AnswerWithSources
from rag.retrieval import Chunk


def _result(answer: str, sources: list[str], citations: list[int]) -> AnswerWithSources:
    chunks = [
        Chunk(content=f"chunk {i}", source=src, metadata={}, distance=0.1)
        for i, src in enumerate(sources, start=1)
    ]
    return AnswerWithSources(
        answer=answer,
        sources=sources,
        chunks=chunks,
        citations=citations,
    )


def test_citation_precision_all_valid():
    r = _result("see [1] and [2]", ["a.md", "b.md"], [1, 2])
    assert citation_precision(r) == 1.0


def test_citation_precision_flags_bad_index():
    r = _result("see [1] and [9]", ["a.md"], [1, 9])
    assert citation_precision(r) == 0.5


def test_citation_precision_empty():
    r = _result("no cites", ["a.md"], [])
    assert citation_precision(r) == 0.0


def test_citation_source_hit():
    r = _result("fact [2]", ["other.md", "data/corpus/pgvector_readme.md"], [2])
    assert citation_source_hit(r, ["pgvector_readme.md"]) is True
    assert citation_source_hit(r, ["missing.md"]) is False
