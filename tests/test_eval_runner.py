"""Integration-style tests for the eval orchestration (all providers mocked)."""

import json

from eval import evaluate
from rag.generation import AnswerWithSources
from rag.retrieval import Chunk


def test_full_eval_retrieves_once_and_writes_report(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is the fact?",
                    "expected_answer": "The fact is one.",
                    "relevant_sources": ["expected.md"],
                }
            ]
        )
    )
    chunks = [
        Chunk(
            content="The fact is one.",
            source="data/corpus/expected.md",
            metadata={},
            distance=0.1,
        )
    ]
    retrieve_calls = 0

    def fake_retrieve(question, k):
        nonlocal retrieve_calls
        retrieve_calls += 1
        assert question == "What is the fact?"
        assert k == 3
        return chunks

    def fake_generate(question, retrieved):
        assert retrieved is chunks
        return AnswerWithSources(
            answer="The fact is one [1].",
            sources=[chunks[0].source],
            chunks=chunks,
            citations=[1],
        )

    monkeypatch.setattr(evaluate, "retrieve", fake_retrieve)
    monkeypatch.setattr(evaluate, "generate_answer", fake_generate)
    monkeypatch.setattr(evaluate, "judge_answer", lambda *_: 0.75)

    exit_code = evaluate.run_eval(questions_path, k=3, output_dir=tmp_path / "results")

    assert exit_code == 0
    assert retrieve_calls == 1
    reports = list((tmp_path / "results").glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["config"]["top_k"] == 3
    assert len(report["questions_sha256"]) == 64
    assert "code_revision" in report
    assert report["summary"]["mrr"] == 1.0
    assert report["summary"]["citation_source_hit_rate"] == 1.0
    assert report["summary"]["answer_quality"] == 0.75


def test_full_eval_skips_judge_when_expected_answer_is_missing(tmp_path, monkeypatch, capsys):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps([{"question": "What is the fact?", "relevant_sources": ["expected.md"]}]),
        encoding="utf-8",
    )
    chunks = [
        Chunk(
            content="The fact is one.",
            source="data/corpus/expected.md",
            metadata={},
            distance=0.1,
        )
    ]
    monkeypatch.setattr(evaluate, "retrieve", lambda _question, k: chunks)
    monkeypatch.setattr(
        evaluate,
        "generate_answer",
        lambda _question, _chunks: AnswerWithSources(
            answer="The fact is one [1].",
            sources=[chunks[0].source],
            chunks=chunks,
            citations=[1],
        ),
    )

    def unexpected_judge(*_args):
        raise AssertionError("judge should not run without an expected answer")

    monkeypatch.setattr(evaluate, "judge_answer", unexpected_judge)

    assert evaluate.run_eval(questions_path, save_artifacts=False) == 0
    assert "Answer quality:      n/a" in capsys.readouterr().out
