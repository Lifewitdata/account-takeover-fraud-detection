.PHONY: install pipeline test sql dashboard all
export PYTHONPATH := src

install:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e . --no-deps

pipeline:
	python -m ato.pipeline

test:
	pytest tests/test_units.py tests/test_outputs.py

sql:
	pytest tests/test_sql_parity.py

dashboard:
	streamlit run dashboard/app.py

all: pipeline test sql
