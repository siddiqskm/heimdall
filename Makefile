.PHONY: lint lint-fix format test test-strict check check-strict

lint:
	poetry run ruff check .

lint-fix:
	poetry run ruff check . --fix

format:
	poetry run ruff format .

# Normal test run
test:
	poetry run pytest -q

# Stop at first failure
test-strict:
	poetry run pytest -q --maxfail=1

# Lint + normal test
check: lint test

# Lint + strict test (no mercy mode)
check-strict: lint test-strict
