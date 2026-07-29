"""Answer generation with citations via Ollama."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from rag.config import settings
from rag.retrieval import Chunk


@dataclass
class AnswerWithSources:
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


class GenerationError(RuntimeError):
    pass


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    if not chunks:
        context = "(No retrieved context.)"
    else:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(f"[{i}] source={chunk.source}\n{chunk.content}")
        context = "\n\n".join(parts)

    return (
        "You are a helpful assistant. Answer the question using only the context "
        "chunks below. Cite sources inline like [1], [2] referring to the chunk "
        "numbers. If the context is insufficient, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def _ollama_chat(prompt: str) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.ollama_chat_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        raise GenerationError(
            f"Ollama chat failed ({exc}). Is Ollama running and did you "
            f"`ollama pull {settings.ollama_chat_model}`?"
        ) from exc

    message = data.get("message") or {}
    content = message.get("content")
    if not content:
        raise GenerationError(f"Unexpected Ollama chat response: {data!r}")
    return content.strip()


def generate_answer(question: str, chunks: list[Chunk]) -> AnswerWithSources:
    prompt = build_prompt(question, chunks)
    answer = _ollama_chat(prompt)
    sources = list(dict.fromkeys(c.source for c in chunks))
    return AnswerWithSources(answer=answer, sources=sources, chunks=chunks)


def judge_answer(question: str, expected: str, actual: str) -> float:
    """LLM-as-judge score in [0, 1] via Ollama."""
    prompt = (
        "You are grading a short answer. Score how well ACTUAL matches EXPECTED "
        "for the QUESTION on a scale from 0.0 to 1.0. Reply with ONLY a number.\n\n"
        f"QUESTION: {question}\n"
        f"EXPECTED: {expected}\n"
        f"ACTUAL: {actual}\n"
        "Score:"
    )
    raw = _ollama_chat(prompt)
    # Parse first float-like token.
    token = raw.replace(",", " ").split()[0] if raw.strip() else "0"
    try:
        score = float(token)
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, score))
