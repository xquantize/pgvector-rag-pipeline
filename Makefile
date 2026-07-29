.PHONY: db-up db-down db-reset install fetch-corpus ingest eval eval-retrieval test fmt check

db-up:      ## start Postgres+pgvector
	docker compose up -d

db-down:    ## stop the database
	docker compose down

db-reset:   ## wipe DB volume and recreate schema from init_db.sql
	docker compose down -v
	docker compose up -d

install:    ## install deps into the current environment
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -e ".[dev]"; \
	else \
		python -m ensurepip --upgrade; \
		python -m pip install -e ".[dev]"; \
	fi

fetch-corpus: ## refresh Postgres/pgvector docs into data/corpus
	PYTHONPATH=src python -m scripts.fetch_corpus

ingest:     ## build the index from ./data/corpus
	PYTHONPATH=src python -m scripts.run_ingest

eval:       ## run the evaluation harness
	PYTHONPATH=src python -m eval.evaluate

eval-retrieval: ## fast retrieval-only eval (no chat/judge calls)
	PYTHONPATH=src python -m eval.evaluate --retrieval-only

test:       ## run unit tests
	pytest

fmt:        ## format & lint
	ruff format . && ruff check --fix .

check:      ## run the same quality checks as CI
	ruff format --check .
	ruff check .
	pytest
