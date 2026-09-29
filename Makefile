PYTHON ?= python
PYTHONPATH := src
DATA_ROOT ?= /data/train
LABELS_XLSX ?= $(DATA_ROOT)/разметка.xlsx
INPUT ?= /data/test.zip
WEIGHTS ?= artifacts
OUT ?= results.csv
EPOCHS ?= 10
WORKERS ?= 1

.PHONY: train train-experts run batch hard-cases benchmark check compile test

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m dexa_ai.training --data-root $(DATA_ROOT) --out-dir $(WEIGHTS) --epochs $(EPOCHS)

train-experts:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m dexa_ai.train_experts --data-root $(DATA_ROOT) --labels-xlsx $(LABELS_XLSX) --out-dir $(WEIGHTS)/experts

run:
	PYTHONPATH=$(PYTHONPATH) uvicorn dexa_ai.api:app --host 0.0.0.0 --port 8000

batch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m dexa_ai.batch --input $(INPUT) --weights $(WEIGHTS) --out-csv $(OUT) --workers $(WORKERS)

hard-cases:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m dexa_ai.make_hard_cases --data-root $(DATA_ROOT) --weights $(WEIGHTS) --out-dir reports/hard_cases

benchmark:
	bash scripts/benchmark_h200.sh /data/benchmark $(WEIGHTS) 200

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src tests

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

check: compile test
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m dexa_ai.batch --input $(INPUT) --weights $(WEIGHTS) --out-csv reports/results_check.csv --workers 1
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m dexa_ai.benchmark --input $(INPUT) --weights $(WEIGHTS) --n 20 --out reports/benchmark_smoke.json
