.PHONY: install install-all ingest ask eval sweep test lint fmt clean

CONFIG ?= configs/default.yaml

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all,dev]"

ingest:
	rag-eval ingest --config $(CONFIG)

ask:
	rag-eval ask "$(Q)" --config $(CONFIG)

eval:
	rag-eval eval --config $(CONFIG)

sweep:
	rag-eval sweep --config $(CONFIG)

test:
	pytest

lint:
	ruff check .

fmt:
	ruff format .

clean:
	rm -rf .chroma runs .pytest_cache .ruff_cache htmlcov .coverage
