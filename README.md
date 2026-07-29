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

- **Ingestion** — load docs → chunk → embed → store in pgvector.
- **Query** — embed question → similarity search → LLM answer with sources.
- **Eval** — run a fixed Q/A set → score retrieval hit rate + answer quality.

## Prerequisites

1. [Docker](https://docs.docker.com/get-docker/) (for Postgres + pgvector)
2. [Ollama](https://ollama.com/download) with models pulled:

```bash
ollama pull nomic-embed-text   # embeddings (768-dim)
ollama pull llama3.2           # chat / LLM-as-judge
```

## Quickstart

```bash
cp .env.example .env           # defaults are local Ollama — no API keys
make db-up                     # start Postgres + pgvector
python -m venv .venv && source .venv/bin/activate
make install                   # install dependencies
# add .txt / .md / .pdf files to ./data, then:
make ingest                    # build the index
make eval                      # run the evaluation harness
```

Edit `eval/test_questions.json` to match your corpus before trusting eval scores.

### Optional: Hugging Face embeddings (local, free)

```bash
pip install -e ".[huggingface]"
```

In `.env`:

```
EMBEDDING_PROVIDER=huggingface
HF_EMBED_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768
```

Keep `EMBEDDING_DIM` (and `VECTOR(n)` in `scripts/init_db.sql`) aligned with the
model. If you change the dimension after the DB was created, recreate the volume:

```bash
docker compose down -v && make db-up
```

## Tech

Postgres + pgvector, Ollama (or local HF) embeddings, Ollama chat for generation
and LLM-as-judge scoring. Python 3.11+, packaged with `pyproject.toml`.

## Results

| Configuration | Retrieval hit rate | Answer quality |
| ------------- | ------------------ | -------------- |
| _baseline_    | _tbd_              | _tbd_          |

_(Fill this in from `make eval`. Show at least one before/after improvement.)_

## What I'd improve next

_(Hybrid search, reranking, larger eval set, metadata filtering, ...)_
