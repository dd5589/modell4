#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TEST_ZIP="${1:?usage: h200_acceptance.sh <test.zip> <benchmark_dir> [n]}"
BENCHMARK_DIR="${2:?usage: h200_acceptance.sh <test.zip> <benchmark_dir> [n]}"
N="${3:-200}"
PORT="${DEXA_H200_PORT:-18000}"
IMAGE="${DEXA_IMAGE:-dexa-ai:final}"
NAME="dexa-ai-h200-acceptance"
OUT_DIR="$ROOT/reports/h200_acceptance"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/health.json "$OUT_DIR"/batch_results.csv "$OUT_DIR"/h200_benchmark.json "$OUT_DIR"/gpu.txt "$OUT_DIR"/docker_image_id.txt

cleanup(){ docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

printf '\n[1/7] target prerequisites\n'
command -v docker >/dev/null || { echo 'ERROR: docker is required on target host'; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'ERROR: nvidia-smi is required on target host'; exit 2; }
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$OUT_DIR/gpu.txt"

printf '\n[2/7] build image\n'
docker build -t "$IMAGE" .
docker image inspect "$IMAGE" --format '{{.Id}}' | tee "$OUT_DIR/docker_image_id.txt"

printf '\n[3/7] start service\n'
docker run --rm -d --name "$NAME" --gpus all -p "$PORT:8000" \
  -v "$ROOT/artifacts:/app/artifacts:ro" \
  "$IMAGE" >/dev/null

printf '\n[4/7] health\n'
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/health" > "$OUT_DIR/health.json"; then break; fi
  sleep 2
done
cat "$OUT_DIR/health.json"
python - <<'PY'
import json
p=json.load(open('reports/h200_acceptance/health.json',encoding='utf8'))
assert p['status']=='ok' and p['model_loaded'] is True
assert p['device']=='cuda'
assert p['deep_members']==3
print('HEALTH PASS')
PY

printf '\n[5/7] real DICOM batch\n'
curl -fsS -X POST -F "file=@$TEST_ZIP" "http://127.0.0.1:$PORT/predict-zip?format=csv" > "$OUT_DIR/batch_results.csv"
python - <<'PY'
import csv
p='reports/h200_acceptance/batch_results.csv'
rows=list(csv.DictReader(open(p,encoding='utf8')))
required={'path_to_study','study_uid','image_uid','anatomical_region','quality_class','violation_type','processing_status','time_of_processing','quality_prob'}
assert rows and required.issubset(rows[0])
allowed_regions={'Поясничный отдел позвоночника','Проксимальный отдел бедра'}
allowed_spine={'Некорректная укладка','Не выравнена ось позвоночника','Присутствуют посторонние предметы'}
allowed_hip={'Некорректная укладка','Некорректная область интереса'}
for r in rows:
  assert r['processing_status']=='Success'
  assert r['anatomical_region'] in allowed_regions
  assert 0 <= float(r['quality_prob']) <= 1
  parts=[x for x in r['violation_type'].split('; ') if x]
  allowed=allowed_spine if r['anatomical_region'].startswith('Поясничный') else allowed_hip
  assert all(x in allowed for x in parts)
print('rows=',len(rows),'success=',len(rows))
print('BATCH PASS')
PY

printf '\n[6/7] H200 inference benchmark inside container\n'
# Benchmark executes inside the same GPU-enabled image used by the API.
docker cp "$BENCHMARK_DIR" "$NAME:/tmp/benchmark_input"
docker exec "$NAME" python -m dexa_ai.benchmark \
  --input /tmp/benchmark_input \
  --weights /app/artifacts \
  --device cuda \
  --warmup 20 \
  --n "$N" \
  --out-json /tmp/h200_benchmark.json

docker cp "$NAME:/tmp/h200_benchmark.json" "$OUT_DIR/h200_benchmark.json"
python - <<'PY'
import json
p=json.load(open('reports/h200_acceptance/h200_benchmark.json',encoding='utf8'))
print(json.dumps(p,ensure_ascii=False,indent=2))
assert p['device']=='cuda'
assert p['n']>=1
assert p['mean_ms']>0 and p['p95_ms']>0
assert p['estimated_3_image_study_p95_ms'] < 180000
print('H200 BENCHMARK PASS')
PY

printf '\n[7/7] package result\n'
cat > "$OUT_DIR/acceptance_summary.json" <<JSON
{
  "status": "PASS",
  "image": "$IMAGE",
  "test_zip": "$TEST_ZIP",
  "benchmark_dir": "$BENCHMARK_DIR",
  "n": $N,
  "generated": "$(date -Iseconds)"
}
JSON
cat "$OUT_DIR/acceptance_summary.json"
printf '\nTARGET H200 ACCEPTANCE PASS\n'
