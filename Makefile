.PHONY: run lint test ci
run:
	uvicorn llm_gateway.main:app --host 0.0.0.0 --port 8000

lint:
	ruff check llm_gateway tests

test:
	pytest -q

ci: lint test
