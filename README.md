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
- **Eval** — fixed grounded Q/A set → retrieval hits, citation checks, LLM judge.

## Corpus

`data/corpus/` is a curated slice of **PostgreSQL 16 docs** plus the **pgvector
README**, fetched by `make fetch-corpus`. Sources and licenses are noted in each
file’s header and in `data/corpus/manifest.json`.

## Prerequisites

1. [Docker](https://docs.docker.com/get-docker/) / OrbStack (for Postgres + pgvector)
2. [Ollama](https://ollama.com/download) with models pulled:

```bash
ollama pull nomic-embed-text   # embeddings (768-dim)
ollama pull llama3.2           # fine as a baseline chat model
# or something stronger, e.g.:
ollama pull qwen2.5:7b
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

| Configuration | Retrieval | Judge quality |
| ------------- | --------- | ------------- |
| hybrid + llama3.2, loose prompt | 100% (20/20) | 0.65 |
| hybrid + qwen2.5, grounded / low-temp prompt | 100% (20/20) | 0.67 |

Retrieval was already solid. Tightening the prompt and swapping chat models
barely moved the LLM-as-judge score — which is a bit expected when the same
local stack is grading itself. Answers *look* cleaner by eye; the judge just
doesn’t capture that well. Eval also tracks citation precision / citation
source hits now so we’re not leaning only on the vibe score.

## What I'd improve next

- Metadata filters (category) at query time
- Reranking top-k before generation
- Vector-only vs hybrid A/B in the Results table
- Tighten expected answers where the judge is being harsh on good replies
