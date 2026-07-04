.PHONY: install install-all ingest ask eval sweep doctor serve test lint fmt \
        docker-build docker-up clean

CONFIG ?= configs/default.yaml

install:
	pip install -e ".[api,dev]"

install-all:
	pip install -e ".[all,dev]"

doctor:
	rag-eval doctor --config $(CONFIG)

ingest:
	rag-eval ingest --config $(CONFIG)

ask:
	rag-eval ask "$(Q)" --config $(CONFIG)

eval:
	rag-eval eval --config $(CONFIG)

sweep:
	rag-eval sweep --config $(CONFIG)

serve:
	rag-eval serve

test:
	pytest

lint:
	ruff check .

fmt:
	ruff format .

docker-build:
	docker build -t rag-eval .

docker-up:
	docker compose up --build

clean:
	rm -rf .chroma runs logs .pytest_cache .ruff_cache htmlcov .coverage
