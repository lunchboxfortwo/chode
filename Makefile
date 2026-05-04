.PHONY: test check test-quick test-server test-engine

test:
	python3 -m pytest tests/test_preflop_encode.py tests/test_postflop_nn.py tests/test_server.py -v --tb=short

test-quick:
	python3 -m pytest tests/test_preflop_encode.py tests/test_postflop_nn.py -v --tb=short

test-server:
	python3 -m pytest tests/test_server.py -v --tb=short

test-engine:
	python3 -m pytest tests/test_engine.py -v --tb=short

check: test
	@echo "✓ All checks passed"
