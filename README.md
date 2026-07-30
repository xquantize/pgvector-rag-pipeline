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
- **Query** — embed question → vector search (or hybrid vector + FTS RRF) → LLM answer.
- **Eval** — fixed grounded Q/A set → retrieval hits, citation checks, LLM judge.

## Corpus

`data/corpus/` is a curated slice of **PostgreSQL 16 docs** plus the **pgvector
README**, fetched by `make fetch-corpus`. Sources and licenses are noted in each
file’s header and in `data/corpus/manifest.json`.

## Prerequisites

1. [Docker](https://docs.docker.com/get-docker/) / OrbStack (for Postgres + pgvector)
2. Python 3.11+
3. [Ollama](https://ollama.com/download) with models pulled:

```bash
ollama pull nomic-embed-text   # embeddings (768-dim)
ollama pull qwen2.5:7b         # default chat and judge model
# Smaller alternative: ollama pull llama3.2, then update OLLAMA_CHAT_MODEL in .env
```

## Quickstart

```bash
cp .env.example .env           # defaults are local Ollama — no API keys
make db-up                     # start Postgres + pgvector
python3.12 -m venv .venv
source .venv/bin/activate
make install
make fetch-corpus              # optional refresh; committed corpus works offline
make ingest                    # upsert chunks from ./data/corpus
make ask Q="How does HNSW differ from IVFFlat?"
make eval                      # 20 grounded questions
make eval-retrieval            # fast: retrieval metrics only, no chat calls
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

Postgres + pgvector (HNSW) + generated `tsvector` for optional hybrid retrieval, Ollama
(or local HF) embeddings, Ollama chat for generation and LLM-as-judge. Python
3.11+, packaged with `pyproject.toml`.

## Results

Retrieval A/B (`nomic-embed-text`, top-5, 20 questions):

| Mode | Hit rate@5 | Precision@1 | MRR | nDCG@5 |
| ---- | ---------- | ----------- | --- | ------ |
| Vector-only | 100% | 80% | 0.900 | 0.926 |
| Hybrid (vector + FTS RRF) | 100% | 80% | 0.900 | 0.926 |

Hybrid moved the BRIN question from rank 2 to 1, moved the multicolumn-index
question from rank 1 to 2, and left the other 18 questions unchanged. It
therefore provides no aggregate retrieval gain over vector-only on the current
corpus and question set.

`vector` is the default on that evidence, since the fused two-scan query costs
more than the single ANN scan and buys nothing measurable here. Hybrid stays
fully supported via `RETRIEVAL_MODE=hybrid` — its one win was a question with a
distinctive literal token (`BRIN`), which is exactly where lexical matching
should help, so this is worth re-running as the corpus grows.

Generation:

| Configuration | Judge quality |
| ------------- | ------------- |
| llama3.2, loose prompt | 0.65 |
| qwen2.5:7b, grounded / low-temp prompt | 0.67 |

Retrieval was already solid. Tightening the prompt and swapping chat models
barely moved the LLM-as-judge score — which is a bit expected when the same
local stack is grading itself. Answers *look* cleaner by eye; the judge just
doesn’t capture that well. Eval also tracks citation precision / citation
source hits now so we’re not leaning only on the vibe score.

### Eval metrics and artifacts

The retrieval report is source-level: repeated chunks from one document count as
one ranked source. It prints:

- **Hit rate@k** — whether any expected source appeared in the top-k.
- **Precision@1** — whether the first source was relevant.
- **MRR** — rewards putting the first relevant source near the top.
- **nDCG@k** — rewards relevant sources appearing higher in the ranking.

Each run writes a full JSON report and a flat per-question CSV under
`eval/results/` (gitignored). The report includes the model, retrieval mode,
chunking settings, top-k, code revision, question-set hash, summary metrics,
and per-question rankings.

```bash
make eval-retrieval                         # quick retrieval iteration
make eval-ab                                # vector vs hybrid, then compare
make eval                                   # full generation + judge run
python -m eval.evaluate --k 10              # try another top-k
python -m eval.evaluate --no-artifacts      # print only
python -m eval.evaluate --output-dir /tmp/eval-runs
```

### Comparing two runs

Saved reports can be diffed to see whether a change actually helped. The
comparison refuses to run unless both reports cover the same question set, so a
"win" can't come from quietly editing the questions.

```bash
make eval-compare-latest                    # diff the two most recent runs
make eval-compare BASE=eval/results/a.json CANDIDATE=eval/results/b.json
```

It reports which config values changed, the delta on each summary metric, and
the questions whose first relevant source moved up or down the ranking.

## What I'd improve next

- Metadata filters (category) at query time
- Reranking top-k before generation
- Tighten expected answers where the judge is being harsh on good replies
