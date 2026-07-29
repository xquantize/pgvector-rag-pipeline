"""Tests for the query entry point used by `make ask`."""

from rag.generation import AnswerWithSources
from scripts import run_query


def test_main_prints_answer_and_sources(monkeypatch, capsys):
    monkeypatch.setattr(
        run_query,
        "answer_question",
        lambda question, k: AnswerWithSources(
            answer=f"Answer to {question}",
            sources=["docs/a.md", "docs/b.md"],
            chunks=[],
            citations=[],
        ),
    )

    exit_code = run_query.main(["What is pgvector?", "-k", "3"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Answer to What is pgvector?" in output
    assert "docs/a.md" in output
    assert "docs/b.md" in output


def test_main_reports_pipeline_errors(monkeypatch, capsys):
    def fail(_question, k):
        del k
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(run_query, "answer_question", fail)

    assert run_query.main(["Question"]) == 1
    assert "query failed: provider unavailable" in capsys.readouterr().err
