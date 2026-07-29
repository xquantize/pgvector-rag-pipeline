"""Embedding model wrapper — Ollama (default) or local Hugging Face."""

from __future__ import annotations

import httpx

from rag.config import settings


class EmbeddingError(RuntimeError):
    pass


def _validate_dim(vectors: list[list[float]]) -> list[list[float]]:
    expected = settings.embedding_dim
    for i, vec in enumerate(vectors):
        if len(vec) != expected:
            raise EmbeddingError(
                f"Embedding dim mismatch at index {i}: got {len(vec)}, "
                f"expected {expected} (EMBEDDING_DIM / VECTOR in init_db.sql)."
            )
    return vectors


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    base = settings.ollama_base_url.rstrip("/")
    try:
        with httpx.Client(timeout=120.0) as client:
            # Prefer batch /api/embed (Ollama ≥0.5).
            resp = client.post(
                f"{base}/api/embed",
                json={"model": settings.ollama_embed_model, "input": texts},
            )
            if resp.status_code == 404:
                # Legacy single-prompt endpoint.
                embeddings: list[list[float]] = []
                for text in texts:
                    legacy = client.post(
                        f"{base}/api/embeddings",
                        json={"model": settings.ollama_embed_model, "prompt": text},
                    )
                    legacy.raise_for_status()
                    emb = legacy.json().get("embedding")
                    if not emb:
                        raise EmbeddingError(
                            f"Unexpected Ollama embeddings response: {legacy.text}"
                        )
                    embeddings.append(emb)
                return _validate_dim(embeddings)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        raise EmbeddingError(
            f"Ollama embed failed ({exc}). Is Ollama running and did you "
            f"`ollama pull {settings.ollama_embed_model}`?"
        ) from exc

    embeddings = data.get("embeddings")
    if not embeddings:
        raise EmbeddingError(f"Unexpected Ollama embed response: {data!r}")
    return _validate_dim(embeddings)


def _embed_huggingface(texts: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            'Hugging Face embeddings require: pip install -e ".[huggingface]"'
        ) from exc

    # HF_TOKEN / HUGGING_FACE_HUB_TOKEN in the environment are picked up automatically.
    model = SentenceTransformer(settings.hf_embed_model)
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    return _validate_dim(vectors)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if settings.embedding_provider == "ollama":
        # Ollama accepts batches; keep batches modest for memory.
        batch_size = 32
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(_embed_ollama(texts[i : i + batch_size]))
        return out
    if settings.embedding_provider == "huggingface":
        return _embed_huggingface(texts)
    raise EmbeddingError(f"Unknown embedding provider: {settings.embedding_provider}")


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
