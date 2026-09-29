# Финальный 3-дневный план сдачи

## День 1 — ML freeze + reproducibility

### Утро
1. Зафиксировать текущий код/веса и `SHA256SUMS.txt`.
2. Не менять архитектуру deep ensemble.
3. Проверить `artifacts/experts/*` и `reports/expert_validation_stratified.json`.

### День 1, середина
Запустить полную локальную проверку:

```bash
export PYTHONPATH=src
export DEXA_TEST_DICOM=/data/train/Исследования/<study>/*.dcm
bash scripts/final_checks.sh /data/train /data/test.zip artifacts
```

Вручную проверить 3 hard cases:

```text
reports/hard_cases/case_axis.png
reports/hard_cases/case_artifact.png
reports/hard_cases/case_roi.png
reports/hard_cases/predictions.json
```

### День 1, вечер
1. Сверить обязательные колонки CSV.
2. Прогнать batch на всём доступном debug/validation archive.
3. Проверить стабильность повторного запуска на одинаковом DICOM.
4. Не использовать debug archive как performance benchmark.

## День 2 — Target H200 / Docker

На каждой целевой VM:

```bash
nvidia-smi -L
docker --version
docker build -t dexa-ai:final .
docker run --rm --gpus all -p 8000:8000 dexa-ai:final
```

Затем:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
bash scripts/benchmark_h200.sh /data/benchmark artifacts 200
```

Сохранить:

```text
artifacts/h200_benchmark.json
```

Обязательные поля: GPU, driver, p50, p95, mean, max, throughput и оценка 3-image study.

### День 2, вечер
Прогнать максимально похожий на финальный пакет DICOM:

```bash
PYTHONPATH=src python -m dexa_ai.batch \
  --input /data/final_test \
  --weights artifacts \
  --out-csv final_submission.csv \
  --workers 1
```

Проверить отсутствие crash/traceback, а все ошибки должны стать `Failure`-строками.

## День 3 — QA + pitch + final package

### Утро
1. Остановить кодовые изменения.
2. Зафиксировать Docker image ID.
3. Зафиксировать веса и SHA256.
4. Запустить финальный `pytest` / `compileall` / API smoke.

### День 3, середина
Прогнать презентацию в следующем порядке:

```text
Problem → TЗ → data/split → architecture → rare experts → metrics → 3 hard cases → backend → H200 plan → limitations → submission
```

### День 3, вечер
Сформировать:

- `dexa_ai_final.zip` — исходники + веса + docs + demo assets;
- `final_release_qa.xlsx`;
- `pitch/dexa_ai_pitch_final.pptx`;
- финальный `final_submission.csv` только для передачи организатору;
- H200 benchmark JSON.

## Если времени осталось < 6 часов

Приоритет:

1. Docker build/health на H200;
2. closed-test batch без падений;
3. CSV schema;
4. pitch rehearsal;
5. только потом косметика.

Не делать:

- большой новый backbone;
- image-level random split;
- новые labels без размеченных данных;
- обещания clinical validation;
- публикацию закрытого теста или медицинских изображений.
