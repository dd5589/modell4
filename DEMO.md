# 5-minute demo script

## 1. Start

CPU/local:

```bash
PYTHONPATH=src uvicorn dexa_ai.api:app --host 0.0.0.0 --port 8000
```

Target GPU:

```bash
docker compose up --build
```

Open `http://localhost:8000`.

## 2. Single study

Upload one DICOM and show:

- `anatomical_region`;
- `quality_class`;
- `quality_prob`;
- `violation_type` in the exact organizer vocabulary;
- `processing_status`;
- `time_of_processing`;
- `review_required`;
- `axis_angle_deg` for spine;
- `quality_prob`.

## 3. Batch

```bash
PYTHONPATH=src python -m dexa_ai.batch \
  --input /data/test.zip \
  --weights artifacts \
  --out-csv results.csv \
  --workers 1
```

The provided debug archive produced 3/3 successful rows in the current environment. These are format/robustness checks, not an independent performance test.

## 4. Hard cases

Show, in this order:

1. `reports/hard_cases/case_axis.png` — spine axis violation;
2. `reports/hard_cases/case_artifact.png` — foreign-object/artifact;
3. `reports/hard_cases/case_roi.png` — hip ROI violation.

Explain that the evidence combines the image, expert signal and the final structured output.

## 5. Metrics

Open `reports/final_release_qa.xlsx`, especially `Validation`, `Runtime Evidence` and `Visible Test`. State explicitly: these are internal study-level holdout results; the final closed-test metrics are calculated by the organizer.

## 6. H200

Do not claim an H200 benchmark until it has actually been run. After target execution, fill `reports/h200_benchmark.json` and update the final slide with real p50/p95/throughput.
