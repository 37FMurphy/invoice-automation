.PHONY: setup db run test lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev,eval]"

db:
	docker compose up -d db

run:
	.venv/bin/uvicorn app.main:app --reload --port 8000

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .
