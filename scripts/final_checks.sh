#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
DATA_ROOT="${1:-/data/train}"
INPUT="${2:-/data/test.zip}"
WEIGHTS="${3:-artifacts}"

printf '\n[1/6] compileall\n'
python -m compileall -q src tests

printf '\n[2/6] pytest\n'
python -m pytest -q

printf '\n[3/6] health/API import\n'
python - <<'PY'
from fastapi.testclient import TestClient
from dexa_ai.api import app
r=TestClient(app).get('/health')
print(r.status_code)
print(r.json())
assert r.status_code == 200
assert r.json().get('model_loaded') is True
PY

printf '\n[4/6] debug batch\n'
python -m dexa_ai.batch --input "$INPUT" --weights "$WEIGHTS" --out-csv reports/final_checks_results.csv --workers 1

printf '\n[5/6] hard cases\n'
if [[ -d "$DATA_ROOT/Исследования" ]]; then
  python -m dexa_ai.make_hard_cases --data-root "$DATA_ROOT" --weights "$WEIGHTS" --out-dir reports/hard_cases
else
  echo "skip: DATA_ROOT/Исследования not found at $DATA_ROOT"
fi

printf '\n[6/6] status summary\n'
python - <<'PY'
import json
from pathlib import Path
p=Path('reports/final_checks_results.csv')
print('batch_csv:', p, 'exists=', p.exists())
if p.exists():
    import csv
    rows=list(csv.DictReader(p.open(encoding='utf-8')))
    print('rows=',len(rows),'success=',sum(r.get('processing_status')=='Success' for r in rows))
print('hard_cases=', list(Path('reports/hard_cases').glob('case_*.png')))
PY
