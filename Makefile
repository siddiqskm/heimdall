.PHONY: lint lint-fix format test test-strict-quite test-strict check check-strict

lint:
	poetry run ruff check .

lint-fix:
	poetry run ruff check . --fix

format:
	poetry run ruff format .

# Normal test run
test:
	poetry run pytest -q

# Stop at first failure in quite mode
test-strict-quite:
	poetry run pytest -q --maxfail=1

# Stop at first failure, with heimdall logging enabled
test-strict:
	poetry run pytest --maxfail=1 --log-heimdall=DEBUG

# Lint + normal test
check: lint test

# Lint + strict test (no mercy mode)
check-strict: lint test-strict
