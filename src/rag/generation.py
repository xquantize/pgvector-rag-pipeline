"""Answer generation with citations via Ollama.

Industry-style grounded generation: system/user split, low temperature,
short factual answers constrained to retrieved context, structured LLM judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx

from rag.config import settings
from rag.retrieval import Chunk

SYSTEM_PROMPT = """\
You are a retrieval-grounded assistant for technical documentation.

Rules:
- Answer using ONLY the provided context chunks.
- Prefer one or two short factual sentences. No preamble, no essays.
- Cite supporting chunks inline with bracket markers like [1] or [2].
- If the context is missing or insufficient, reply exactly: I don't know based on the provided context.
- Never invent operators, APIs, defaults, or numbers that are not in the context.
- If sources conflict, say so briefly and cite both.
"""

JUDGE_SYSTEM_PROMPT = """\
You grade short answers for factual agreement with a reference answer.
Ignore style, length, and citation markers. Focus on whether the key fact is correct.
Respond with JSON only: {"score": <number from 0.0 to 1.0>, "reason": "<one short sentence>"}
"""


@dataclass
class AnswerWithSources:
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)


class GenerationError(RuntimeError):
    pass


def build_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(No retrieved context.)"
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        title = (chunk.metadata or {}).get("title") or chunk.source
        # Cap each chunk so the prompt stays focused (common production guardrail).
        body = chunk.content.strip()
        max_chars = settings.context_chunk_chars
        if len(body) > max_chars:
            body = body[: max_chars - 1].rstrip() + "…"
        parts.append(f"[{i}] title={title}\nsource={chunk.source}\n{body}")
    return "\n\n".join(parts)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    user = (
        f"Context:\n{build_context(chunks)}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer briefly with citations:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Single-string prompt (tests / debugging). Prefer build_messages for chat."""
    messages = build_messages(question, chunks)
    return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)


def extract_citations(answer: str) -> list[int]:
    found = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    return list(dict.fromkeys(found))


def _ollama_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    num_predict: int | None = None,
) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.ollama_chat_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "num_predict": settings.llm_num_predict if num_predict is None else num_predict,
        },
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
    answer = _ollama_chat(build_messages(question, chunks))
    sources = list(dict.fromkeys(c.source for c in chunks))
    return AnswerWithSources(
        answer=answer,
        sources=sources,
        chunks=chunks,
        citations=extract_citations(answer),
    )


def parse_judge_score(raw: str) -> float:
    """Parse judge output: JSON preferred, else first float token."""
    text = raw.strip()
    # Fenced JSON
    fenced = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(0))
            score = float(data["score"])
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    try:
        return max(0.0, min(1.0, float(match.group(0))))
    except ValueError:
        return 0.0


def judge_answer(question: str, expected: str, actual: str) -> float:
    """LLM-as-judge score in [0, 1] via Ollama."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION: {question}\n"
                f"EXPECTED: {expected}\n"
                f"ACTUAL: {actual}\n"
            ),
        },
    ]
    raw = _ollama_chat(
        messages,
        temperature=settings.judge_temperature,
        num_predict=80,
    )
    return parse_judge_score(raw)
