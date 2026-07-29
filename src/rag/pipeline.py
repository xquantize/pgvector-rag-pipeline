"""End-to-end query pipeline: question in, cited answer out."""

from __future__ import annotations

from rag.generation import AnswerWithSources, generate_answer
from rag.retrieval import retrieve


def answer_question(question: str, k: int | None = None) -> AnswerWithSources:
    chunks = retrieve(question, k=k)
    return generate_answer(question, chunks)
