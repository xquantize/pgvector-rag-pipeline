# pgvector-rag-pipeline

RAG over Postgres/pgvector, with a real evaluation harness.

Ask natural-language questions over a document corpus. Documents are chunked,
embedded, and stored in Postgres via the pgvector extension; a query retrieves
the most relevant chunks and an LLM answers with citations. A small evaluation
harness measures retrieval and answer quality so pipeline changes can be
compared with numbers rather than vibes.

**No paid OpenAI/Anthropic keys.** Embeddings and generation use local
[Ollama](https://ollama.com) by default. Optional local Hugging Face embeddings
via `sentence-transformers`.

## Architecture

- **Ingestion** — load docs → chunk → embed → upsert into pgvector (idempotent).
- **Query** — embed question → hybrid search (vector + FTS RRF in SQL) → LLM answer.
- **Eval** — fixed grounded Q/A set → retrieval hit rate + answer quality.

## Corpus

`data/corpus/` is a curated slice of **PostgreSQL 16 docs** plus the **pgvector
README**, fetched by `make fetch-corpus`. Sources and licenses are noted in each
file’s header and in `data/corpus/manifest.json`.

## Prerequisites

1. [Docker](https://docs.docker.com/get-docker/) / OrbStack (for Postgres + pgvector)
2. [Ollama](https://ollama.com/download) with models pulled:

```bash
ollama pull nomic-embed-text   # embeddings (768-dim)
ollama pull llama3.2           # chat / LLM-as-judge
```

## Quickstart

```bash
cp .env.example .env           # defaults are local Ollama — no API keys
make db-up                     # start Postgres + pgvector
source .venv/bin/activate      # Python >= 3.11
make install
make fetch-corpus              # optional refresh; committed corpus works offline
make ingest                    # upsert chunks from ./data/corpus
make eval                      # 20 grounded questions
```

Schema changes require a fresh volume: `make db-reset`.

### Optional: Hugging Face embeddings (local, free)

```bash
uv pip install -e ".[huggingface]"
```

In `.env`:

```
EMBEDDING_PROVIDER=huggingface
HF_EMBED_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768
```

## Tech

Postgres + pgvector (HNSW) + generated `tsvector` for hybrid retrieval, Ollama
(or local HF) embeddings, Ollama chat for generation and LLM-as-judge. Python
3.11+, packaged with `pyproject.toml`.

## Results

| Configuration | Retrieval hit rate | Answer quality |
| ------------- | ------------------ | -------------- |
| hybrid + nomic-embed-text + llama3.2 | 100% (20/20) | 0.65 |

Measured with `make eval` on the committed Postgres/pgvector corpus. Retrieval
is strong; answer quality is the gap (local chat model + LLM-as-judge noise).

## What I'd improve next

- Tighter generation prompt / stronger local chat model (lift answer quality)
- Metadata filters (category) at query time
- Reranking top-k before generation
- Citation-accuracy metric alongside judge score
