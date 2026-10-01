.PHONY: setup db test lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

db:
	docker compose up -d db

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .
