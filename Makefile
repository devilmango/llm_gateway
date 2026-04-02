.PHONY: run
run:
	uvicorn llm_gateway.main:app --host 0.0.0.0 --port 8000
