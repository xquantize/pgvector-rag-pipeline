.PHONY: db-up db-down install ingest eval test fmt

db-up:      ## start Postgres+pgvector
	docker compose up -d

db-down:    ## stop the database
	docker compose down

install:    ## install deps into the current environment
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -e ".[dev]"; \
	else \
		python -m ensurepip --upgrade; \
		python -m pip install -e ".[dev]"; \
	fi

ingest:     ## build the index from ./data
	PYTHONPATH=src python -m scripts.run_ingest

eval:       ## run the evaluation harness
	PYTHONPATH=src python -m eval.evaluate

test:       ## run unit tests
	pytest

fmt:        ## format & lint
	ruff format . && ruff check --fix .
