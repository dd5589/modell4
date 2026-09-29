# H200 acceptance run

This run must be executed on the target Linux VM because the current authoring runtime has no Docker engine and no NVIDIA H200 device.

## 0. Preconditions

Expected target from the competition specification: 44 CPU, 256 GB RAM, H200 141 GB VRAM per VM. The final test environment may expose one H200 to the container; verify with `nvidia-smi`.

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version
docker --version
```

Stop if `nvidia-smi -L` or `docker --version` fails.

## 1. Build

```bash
docker build -t dexa-ai:final .
```

Optional: save image digest for reproducibility:

```bash
docker image inspect dexa-ai:final --format '{{.Id}}'
```

## 2. Start and health

```bash
docker run --rm --gpus all -p 8000:8000 dexa-ai:final
```

In a second shell:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Expected fields include:

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "deep_members": 3,
  "expert_members": [
    "hip_roi",
    "hip_rotation",
    "spine_artifact",
    "spine_placement"
  ]
}
```

## 3. Single-DICOM smoke test

```bash
curl -s -X POST \
  -F 'file=@/data/example.dcm' \
  'http://127.0.0.1:8000/predict?explain=true' | python -m json.tool
```

Confirm `processing_status=Success` and a supported `anatomical_region`.

## 4. Batch acceptance

```bash
curl -s -X POST \
  -F 'file=@/data/test.zip' \
  'http://127.0.0.1:8000/predict-zip?format=csv' > results.csv
```

Check required columns:

```bash
python - <<'PY'
import csv
r=list(csv.DictReader(open('results.csv',encoding='utf-8')))
required={'path_to_study','study_uid','image_uid','anatomical_region','quality_class','violation_type','processing_status','time_of_processing'}
assert r and required.issubset(r[0])
assert all(x['processing_status'] in {'Success','Failure'} for x in r)
print('rows=',len(r),'success=',sum(x['processing_status']=='Success' for x in r))
PY
```

## 5. H200 benchmark

```bash
bash scripts/benchmark_h200.sh /data/benchmark artifacts 200
```

Keep the output and fill:

```text
reports/h200_benchmark.json
```

Use `reports/h200_benchmark.template.json` as the schema template.

Record GPU name, driver, p50/p95/mean/max latency, throughput and GPU memory.

## 6. Closed-test run

Only after the health and benchmark checks pass:

```bash
PYTHONPATH=src python -m dexa_ai.batch \
  --input /data/final_test \
  --weights artifacts \
  --out-csv final_submission.csv \
  --workers 1
```

Do not put the closed test archive into public repositories or the final source archive.
