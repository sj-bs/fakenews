.PHONY: install
install:
	poetry install

.PHONY: update
update:
	poetry update

.PHONY: lint
lint:
	poetry run ruff format --check .

.PHONY: format
format:
	poetry run ruff format .

.PHONY: sort
sort:
	poetry run isort .

.PHONY: run
run:
	poetry run python -m app.main

