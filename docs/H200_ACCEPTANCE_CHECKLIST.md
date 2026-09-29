# H200 acceptance checklist

This is the final target-host acceptance run for the frozen bundle. The competition target specifies one H200 141 GB VM with 44 CPU and 256 GB RAM per VM.

## Before the run

```bash
nvidia-smi -L
docker --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

The expected GPU is H200 141 GB. A missing `nvidia-smi` or Docker is a host/setup issue, not a model failure.

## One-command acceptance

From the project root:

```bash
bash scripts/h200_acceptance.sh /data/Для\ теста.zip /data/benchmark 200
```

The script performs:

1. host GPU/Docker checks;
2. Docker build;
3. GPU-enabled service startup;
4. `/health` assertion (`device=cuda`, 3 deep members);
5. real DICOM ZIP through `/predict-zip` and schema validation;
6. in-container CUDA benchmark with warmup 20 and N=200;
7. 3-image-study latency gate against the 180 s specification;
8. persistence of all evidence under `reports/h200_acceptance/`.

## Evidence files

```text
reports/h200_acceptance/
  gpu.txt
  docker_image_id.txt
  health.json
  batch_results.csv
  h200_benchmark.json
  acceptance_summary.json
```

## API single-DICOM check

When a standalone DICOM is available:

```bash
curl -fsS -X POST \
  -F 'file=@/data/example.dcm' \
  'http://127.0.0.1:18000/predict?explain=true' | python -m json.tool
```

Expected: `processing_status=Success`, `quality_prob` in `[0,1]`, and organizer-defined `anatomical_region` string.

## Important interpretation

The authoring runtime has no Docker daemon and no NVIDIA GPU. Therefore the H200 fields remain `PENDING_TARGET_HOST` until this exact script is executed on the target VM. Do not substitute CPU numbers for the H200 benchmark in the final presentation.
