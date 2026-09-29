# FINAL 72H RUNBOOK — что делать после текущего пакета

## Цель
Зафиксировать то, что уже работает, и не ломать рабочий MVP перед финальной отправкой. Три приоритетных блока: rare-label experts, H200 benchmark, hard-case demo.

## Блок A — rare-label experts

### 1. Запустить
```bash
PYTHONPATH=src python -m dexa_ai.train_experts \
  --data-root /data/train \
  --labels-xlsx /data/train/разметка.xlsx \
  --out-dir artifacts/experts
```

### 2. Проверить артефакты
```text
artifacts/experts/
  spine_placement.pkl
  spine_artifact.pkl
  hip_rotation.pkl
  hip_roi.pkl
  expert_config.json
  expert_oof.json        # optional, if cross-validation finished
```

### 3. Что показывать на защите
- `axis_angle_deg` как объяснимый геометрический признак;
- `expert_probabilities` как дополнительный компонент ансамбля;
- `review_required` как механизм human-in-the-loop для пограничных случаев.

Не называть эти компоненты «клинически валидированными».

## Блок B — H200

Target environment from competition specification:
- 2 × H200 141 GB;
- 44 CPU;
- 256 GB RAM;
- local Docker execution.

### 1. Build
```bash
docker build -t dexa-ai:final .
```

### 2. Health
```bash
docker run --rm --gpus all -p 8000:8000 dexa-ai:final
curl http://localhost:8000/health
```
Expected:
```json
{"status":"ok","model_loaded":true}
```

### 3. GPU benchmark
В отдельной shell-сессии контейнера/хоста:
```bash
bash scripts/benchmark_h200.sh /data/benchmark artifacts 200
```

Зафиксировать:
- GPU model;
- driver;
- p50/p95/mean/max latency;
- images/sec;
- RAM/VRAM;
- долю успешной обработки.

### 4. Пройти batch acceptance
```bash
PYTHONPATH=src python -m dexa_ai.batch \
  --input /data/final_test \
  --weights artifacts \
  --out-csv results.csv \
  --workers 1
```
Проверить:
- нет необработанных exception;
- каждая строка имеет `processing_status`;
- число DICOM == число строк, кроме intentionally skipped unsupported files;
- все required columns присутствуют.

## Блок C — hard-case demo

```bash
bash scripts/make_hard_cases.sh /data/train artifacts reports/hard_cases
```

Кейс 1: spine axis.
Кейс 2: spine artifact.
Кейс 3: hip ROI.

На каждом слайде показать original image + evidence + output JSON/CSV row.

## Блок D — final checklist

### Mandatory
- trained weights;
- DICOM reader;
- binary quality;
- spine + proximal femur;
- batch;
- CSV/XLSX-compatible output;
- API;
- Docker;
- README/docs;
- presentation;
- error handling.

### Optional / partial
- segmentation/keypoints;
- DICOM SR;
- automatic correction with human confirmation.

## Блок E — что не делать

Не переключать архитектуру на тяжёлый transformer за день до сдачи.

Не делать random image-level split.

Не выдавать debug/test archive за независимый benchmark.

Не скрывать отсутствие H200 run в текущей среде.

Не обещать side-aware hip classification при отсутствии надёжного label/metadata источника.
