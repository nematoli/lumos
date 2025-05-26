.PHONY: format lint check-format install-dev

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Format code with black and isort
format:
	black .
	isort .

# Check if code is formatted correctly (without making changes)
check-format:
	black --check .
	isort --check-only .

# Run linting
lint:
	flake8 .

# Run all checks (format check + lint)
check: check-format lint

# Format and then run checks
format-and-check: format check 