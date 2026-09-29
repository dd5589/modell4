# API

Base URL по умолчанию:

```text
http://127.0.0.1:8000
```

Swagger/OpenAPI:

```text
GET /docs
```

## GET /health

Возвращает статус загрузки модели и устройства.

## POST /predict

Multipart upload одного DICOM.

Пример:

```bash
curl -X POST -F "file=@case.dcm" http://127.0.0.1:8000/predict
```

## POST /predict-zip

Multipart upload ZIP с DICOM.

JSON:

```text
POST /predict-zip?format=json
```

CSV:

```text
POST /predict-zip?format=csv
```

## Output contract

```text
path_to_study
study_uid
image_uid
anatomical_region
quality_class
violation_type
processing_status
time_of_processing
quality_prob
```
