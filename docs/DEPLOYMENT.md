# Deployment

## Local CPU

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
PYTHONPATH=src uvicorn dexa_ai.api:app --host 0.0.0.0 --port 8000
```

## Local GPU / H200

The target competition environment specifies 2 × H200 141 GB, 44 CPU and 256 GB RAM per VM. The container is CUDA-enabled and can use the host GPU via `--gpus all`.

```bash
docker build -t dexa-ai:final .
docker run --rm --gpus all -p 8000:8000 dexa-ai:final
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Expected health response includes `status=ok`, `model_loaded=true`, `device=cuda` and the loaded member counts.

The final CSV contract contains exactly these columns:
`path_to_study`, `study_uid`, `image_uid`, `anatomical_region`, `quality_class`, `violation_type`, `processing_status`, `time_of_processing`, `quality_prob`.

`anatomical_region` and `violation_type` use the exact organizer-defined Russian strings documented in `README.md`.

## Batch contract

```bash
curl -s -X POST \
  -F 'file=@/data/test.zip' \
  'http://127.0.0.1:8000/predict-zip?format=csv' > results.csv
```

Required output columns are documented in `README.md`.

## Acceptance checklist

- no unhandled exceptions;
- every input produces a row or an explicit `Failure` row;
- `study_uid` and `image_uid` preserved;
- processing time recorded;
- repeated runs on identical files produce stable predictions;
- medical images stay local to the host/network;
- the hidden closed test is run only by/on the final target environment; the participant-visible `Для теста.zip` is treated as a debug/format test only.
