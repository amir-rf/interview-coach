# Makefile for Voice Interview Coach

.PHONY: install playground run smoke-trace generate-traces grade frontend

install:
	agents-cli install

playground:
	agents-cli playground

run:
	agents-cli run "Start mock interview"

smoke-trace:
	uv run python3 tests/eval/generate_traces.py --case malicious_resume

generate-traces:
	uv run python3 tests/eval/generate_traces.py

grade:
	agents-cli eval grade --config tests/eval/eval_config.yaml --traces artifacts/traces/generated_traces.json

frontend:
	uv run uvicorn frontend.main:app --host 127.0.0.1 --port 8090
