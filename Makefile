all: lint pytest-fast
black:
	-@black -i -r torchsisr tests
isort:
	-@isort torchsisr tests
pylint:
	-@pylint --exit-zero --ignore-patterns "flycheck_*" torchsisr/*.py bin/*.py tests/*.py
mypy:
	-@mypy torchsisr/*.py bin/*.py tests/*.py
pyupgrade:
	-@find torchsisr -type f -name "*.py" -print |xargs pyupgrade --py310-plus
autowalrus:
	-@find torchsisr -type f -name "*.py" -print |xargs auto-walrus
refurb:
	-@refurb tests/
	-@refurb torchsisr/
ruffix:
	-@ruff check torchsisr tests --fix --output-format pylint
ruff:
	-@ruff check torchsisr tests --output-format pylint

pytest-fast:
	-@pytest --cov -m 'not requires_dataset'
pytest-full:
	-@pytest --cov
precommit:
	-@pre-commit



lint: pylint ruff mypy refurb

fix: ruffix pyupgrade autowalrus isort
