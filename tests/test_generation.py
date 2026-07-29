"""Unit tests for grounded generation helpers (no Ollama required)."""

from rag.generation import (
    build_messages,
    build_prompt,
    extract_citations,
    parse_judge_score,
)
from rag.retrieval import Chunk


def _chunk(text: str, source: str = "data/corpus/demo.md") -> Chunk:
    return Chunk(content=text, source=source, metadata={"title": "Demo"}, distance=0.1)


def test_build_messages_uses_system_and_user_roles():
    messages = build_messages("What is X?", [_chunk("X is 1.")])
    assert messages[0]["role"] == "system"
    assert "ONLY the provided context" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "[1]" in messages[1]["content"]
    assert "What is X?" in messages[1]["content"]


def test_build_prompt_includes_context_and_question():
    prompt = build_prompt("What is X?", [_chunk("X is 1.")])
    assert "SYSTEM:" in prompt
    assert "USER:" in prompt
    assert "X is 1." in prompt


def test_extract_citations_dedupes_and_orders_by_appearance():
    assert extract_citations("See [2] and also [1] plus [2] again.") == [2, 1]


def test_parse_judge_score_prefers_json():
    assert parse_judge_score('{"score": 0.8, "reason": "close"}') == 0.8


def test_parse_judge_score_falls_back_to_number():
    assert parse_judge_score("Score: 0.55 (pretty good)") == 0.55


def test_parse_judge_score_clamps():
    assert parse_judge_score('{"score": 1.5}') == 1.0
    assert parse_judge_score('{"score": -0.2}') == 0.0
