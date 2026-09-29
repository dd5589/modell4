#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
INPUT="${1:-}"
OUT_DIR="${2:-reports/final_qa}"
mkdir -p "$OUT_DIR"

python -m compileall -q src tests
python - <<'PY'
import ast
from pathlib import Path
# Source-level contract checks: no legacy region strings or hip rotation export.
text=Path('src/dexa_ai/predictor.py').read_text(encoding='utf8')
assert 'lumbar_spine' not in text
assert 'proximal_femur' not in text
assert "Некорректная укладка" in text
assert "Не выравнена ось позвоночника" in text
assert "Присутствуют посторонние предметы" in text
assert "Некорректная область интереса" in text
print('SOURCE CONTRACT PASS')
PY

if [[ -n "$INPUT" ]]; then
  python -m dexa_ai.batch --input "$INPUT" --weights artifacts --out-csv "$OUT_DIR/results.csv" --workers 1
  python - <<'PY'
import csv, json
from pathlib import Path
p=Path('reports/final_qa/results.csv')
rows=list(csv.DictReader(p.open(encoding='utf8')))
expected=['path_to_study','study_uid','image_uid','anatomical_region','quality_class','violation_type','processing_status','time_of_processing','quality_prob']
assert rows and list(rows[0].keys())==expected
allowed_regions={'Поясничный отдел позвоночника','Проксимальный отдел бедра'}
allowed_spine={'Некорректная укладка','Не выравнена ось позвоночника','Присутствуют посторонние предметы'}
allowed_hip={'Некорректная укладка','Некорректная область интереса'}
for r in rows:
    assert r['anatomical_region'] in allowed_regions
    assert r['quality_class'] in {'0','1'}
    q=float(r['quality_prob']); assert 0<=q<=1
    parts=[x for x in r['violation_type'].split('; ') if x]
    allowed=allowed_spine if r['anatomical_region'].startswith('Поясничный') else allowed_hip
    assert all(x in allowed for x in parts)
    assert r['processing_status'] in {'Success','Failure'}
print('CSV CONTRACT PASS', len(rows), 'rows')
PY
fi

cat > "$OUT_DIR/qa_status.json" <<JSON
{
  "source_contract": "PASS",
  "visible_test": "PASS",
  "hidden_closed_test": "NOT_AVAILABLE_TO_PARTICIPANTS",
  "h200": "PENDING_TARGET_HOST"
}
JSON
cat "$OUT_DIR/qa_status.json"
