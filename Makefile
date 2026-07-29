.PHONY: db-up db-down db-reset install fetch-corpus ingest ask eval eval-retrieval eval-compare eval-compare-latest test fmt check

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

ask:        ## ask the corpus a question: make ask Q="how do HNSW and IVFFlat differ?"
	@if [ -z "$(Q)" ]; then \
		echo 'Usage: make ask Q="your question"'; \
		exit 2; \
	fi
	@PYTHONPATH=src python -m scripts.run_query "$(Q)"

eval:       ## run the evaluation harness
	PYTHONPATH=src python -m eval.evaluate

eval-retrieval: ## fast retrieval-only eval (no chat/judge calls)
	PYTHONPATH=src python -m eval.evaluate --retrieval-only

eval-compare: ## diff two runs: make eval-compare BASE=old.json CANDIDATE=new.json
	@if [ -z "$(BASE)" ] || [ -z "$(CANDIDATE)" ]; then \
		echo "Usage: make eval-compare BASE=path/to/old.json CANDIDATE=path/to/new.json"; \
		echo "Or compare the newest two with: make eval-compare-latest"; \
		exit 2; \
	fi
	PYTHONPATH=src python -m eval.compare "$(BASE)" "$(CANDIDATE)"

eval-compare-latest: ## diff the two most recent runs automatically
	@PYTHONPATH=src python -m eval.compare --latest

test:       ## run unit tests
	python -m pytest

fmt:        ## format & lint
	python -m ruff format . && python -m ruff check --fix .

check:      ## run the same quality checks as CI
	python -m ruff format --check .
	python -m ruff check .
	python -m pytest
