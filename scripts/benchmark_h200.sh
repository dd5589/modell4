#!/usr/bin/env bash
set -euo pipefail
INPUT="${1:?usage: benchmark_h200.sh <dicom-dir> [weights-dir] [n]}"
WEIGHTS="${2:-artifacts}"
N="${3:-200}"

echo '=== GPU ==='
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo '=== Model benchmark ==='
PYTHONPATH=src python -m dexa_ai.benchmark --input "$INPUT" --weights "$WEIGHTS" --device cuda --warmup 20 --n "$N" --out-json artifacts/h200_benchmark.json
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1)
GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
GPU_MEM=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
DRIVER=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
python - <<'PY' "$GPU_NAME" "$GPU_MEM" "$DRIVER"
import json,sys
p='artifacts/h200_benchmark.json'
d=json.load(open(p,encoding='utf-8'))
d.update({'gpu_name':sys.argv[1],'gpu_memory_total':sys.argv[2],'driver_version':sys.argv[3]})
json.dump(d,open(p,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
PY
cat artifacts/h200_benchmark.json
