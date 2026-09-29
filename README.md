# DXA Quality AI — «Танкисты 2.0»

Локальный сервис искусственного интеллекта для автоматизированного контроля качества DEXA/денситометрических исследований в формате DICOM.

Проект разработан под конкурсное ТЗ и поддерживает две обязательные анатомические области:

- **Поясничный отдел позвоночника**
- **Проксимальный отдел бедра**

Сервис определяет качество исследования, типы нарушений, вероятность нарушения и формирует структурированный CSV/JSON-результат. Решение работает локально, без передачи медицинских изображений во внешние сервисы.

> **Статус release:** production-ready local MVP / competition submission package. H200 benchmark и закрытый тест выполняются на инфраструктуре организатора/целевой H200 VM, поскольку эти ресурсы отсутствуют в репозитории и в локальном Windows-окружении разработчика.

## 1. Архитектура

Текущий production inference использует **гетерогенный ансамбль**, а не одну нейросеть:

```text
DICOM / ZIP
    │
    ▼
DICOM reader + preprocessing
    │
    ▼
Region detection
    │
    ├───────────────────────────────────────────────┐
    │                                               │
    ▼                                               ▼
Deep image ensemble                         Feature / expert ensemble
- FastCNN                                   - HOG / LBP / edge features
- DilatedCNN                                - LightGBM bagging
- DepthwiseCNN                              - placement expert
                                            - axis geometry expert (5°)
                                            - artifact expert
                                            - hip rotation expert
                                            - hip ROI expert
    │                                               │
    └───────────────────┬───────────────────────────┘
                        ▼
              Target-specific blending
                        │
                        ▼
                  quality_prob
                        │
                        ▼
                  quality_class
                        │
                        ▼
                 violation_type
```

Для редких и геометрически определимых нарушений используются task-specific experts. Это позволяет не полагаться только на CNN при малом числе положительных примеров.

### Экспериментальный v4

В `experimental/pretrained_v4/` находится отдельный training pipeline для следующего этапа экспериментов: ResNet-50, DenseNet-121, EfficientNet-B0 и ConvNeXt-Tiny с последующим сравнением по study-level OOF/holdout.

**Важно:** пока для этих моделей не получены честные target-host training/validation результаты, они не включаются в production predictor и не должны заявляться как достигший прироста качества вариант.

## 2. Требования к выходу

### `anatomical_region`

Допустимы только два значения:

```text
Поясничный отдел позвоночника
Проксимальный отдел бедра
```

Сторона бедра (лево/право) в итоговом значении не указывается.

### `violation_type`

Для поясничного отдела:

```text
Некорректная укладка
Не выравнена ось позвоночника
Присутствуют посторонние предметы
```

Для проксимального отдела бедра:

```text
Некорректная укладка
Некорректная область интереса
```

Несколько нарушений разделяются строкой `; `.

Пример:

```text
Некорректная укладка; Присутствуют посторонние предметы
```

Если нарушений нет, поле `violation_type` остаётся пустым.

Ротация бедра используется как внутренний признак корректности укладки и не выводится отдельным значением `violation_type`.

### `quality_class`

```text
0 = качественное исследование
1 = есть нарушение
```

### `quality_prob`

Вероятность наличия нарушения в диапазоне `[0, 1]`.

### Обязательные поля результата

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

## 3. DICOM и физический масштаб

Для данных конкурса организатор указал единый масштаб сканера:

```text
X = 0.60 mm/pixel
Y = 1.05 mm/pixel
```

Он используется для физических геометрических критериев.

Для ROI:

```text
2 cm / 0.60 mm ≈ 33 px по X
3 cm / 1.05 mm ≈ 29 px по Y
```

Для оси позвоночника используется критерий допустимого наклона до **5°**.

## 4. Структура репозитория

```text
src/dexa_ai/              # inference, DICOM, API, models, experts, metrics
artifacts/                # frozen production weights and ensemble artifacts
scripts/                  # training, batch, benchmark, QA and acceptance scripts
docs/                     # architecture, API, deployment, training, QA, organizer contract
reports/                  # validation, benchmark, hard cases and QA evidence
windows/                  # Windows 10 helper scripts
pitch/                    # presentation PDF
experimental/pretrained_v4/ # next-generation pretrained training pipeline
tests/                    # smoke/regression tests
Dockerfile                # local/container deployment

docker-compose.yml
requirements.txt
requirements-cpu.txt
```

## 5. Быстрый запуск на Windows 10

### Вариант A — Python/CPU

Нужен Python 3.11 x64.

1. Распакуйте репозиторий, например в:

```text
C:\DXA_AI\
```

2. Запустите:

```text
START_WINDOWS.bat
```

3. Выберите:

```text
1. Install / setup Python environment
2. Start API
3. Run visible test
4. Run manual gold audit
```

Подробная инструкция:

```text
windows/README_WINDOWS10.md
```

### Вариант B — вручную

PowerShell:

```powershell
cd C:\DXA_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
$env:PYTHONPATH = "$PWD\src"
python -m compileall -q src tests
pytest -q
```

Запуск API:

```powershell
$env:PYTHONPATH = "$PWD\src"
uvicorn dexa_ai.api:app --host 0.0.0.0 --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

## 6. Проверка на тестовом ZIP

Команда batch:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m dexa_ai.batch `
  --input "C:\path\to\Для теста.zip" `
  --weights artifacts `
  --out-csv results.csv `
  --workers 1
```

Для Windows также можно использовать:

```text
windows/3_RUN_VISIBLE_TEST_WINDOWS.bat
```

Результат будет в CSV с контрактом организатора.

## 7. Ручная проверка качества модели

Для study-level gold audit подготовлены:

```text
scripts/manual_validation.py
windows/4_RUN_MANUAL_AUDIT_WINDOWS.bat
```

Проверяйте прежде всего:

1. `anatomical_region`
2. `quality_class`
3. `quality_prob`
4. `violation_type`
5. для spine — соответствие правила оси 5°
6. для spine — наличие Th12 / iliac crests
7. для spine — посторонние предметы
8. для hip — большой вертел / шейка / седалищная кость
9. для hip — ротацию как часть `Некорректная укладка`
10. для hip — ROI coverage по физическому масштабу

## 8. API

### `GET /health`

Проверка готовности сервиса.

### `POST /predict`

Один DICOM-файл.

Пример через Swagger: `http://127.0.0.1:8000/docs`.

### `POST /predict-zip?format=json`

Batch ZIP.

### `POST /predict-zip?format=csv`

Batch ZIP → CSV.

### Поведение при ошибках

Ошибка одного файла не должна останавливать весь batch. Такой файл получает:

```text
processing_status = Failure
```

Все результаты должны сохраняться в итоговой таблице.

## 9. Docker

Сборка:

```bash
docker build -t dexa-ai:final .
```

Запуск:

```bash
docker run --rm -p 8000:8000 dexa-ai:final
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Для GPU/H200:

```bash
docker run --rm --gpus all -p 8000:8000 dexa-ai:final
```

## 10. Обучение production baseline/ensemble

Подготовка исходного архива:

```bash
python scripts/prepare_dataset.py \
  --training-zip /path/to/НД_для_обучения.zip \
  --out-dir data/prepared
```

Обучение CNN:

```bash
PYTHONPATH=src bash scripts/train.sh /path/to/unpacked/training
```

Обучение task-specific experts:

```bash
PYTHONPATH=src bash scripts/train_experts.sh \
  /path/to/unpacked/training \
  /path/to/unpacked/training/разметка.xlsx
```

Обучение LightGBM bagging:

```bash
PYTHONPATH=src bash scripts/train_advanced_stack.sh \
  /path/to/unpacked/training \
  /path/to/unpacked/training/разметка.xlsx
```

Подробности:

```text
docs/TRAINING.md
docs/ADVANCED_ENSEMBLE.md
```

## 11. H200 acceptance

На целевой H200 VM выполнить:

```bash
nvidia-smi -L

docker --version

bash scripts/h200_acceptance.sh \
  /data/Для\ теста.zip \
  /data/benchmark \
  200
```

Скрипт выполняет:

- проверку NVIDIA/Docker;
- сборку контейнера;
- запуск GPU-сервиса;
- `/health`;
- batch inference на DICOM ZIP;
- benchmark;
- запись acceptance evidence.

Evidence:

```text
reports/h200_acceptance/
```

Формальный лимит ТЗ — не более 3 минут на одно исследование.

## 12. Closed test организатора

Закрытый тестовый набор не входит в репозиторий.

После предоставления организатором закрытого набора или запуска на их инфраструктуре используется тот же batch-контур:

```bash
PYTHONPATH=src python -m dexa_ai.batch \
  --input /data/CLOSED_TEST.zip \
  --weights artifacts \
  --out-csv closed_test_results.csv \
  --workers 1
```

По закрытому набору участник не должен придумывать собственные ground-truth метрики. Итоговые метрики рассчитываются организатором на их эталонной разметке.

## 13. QA / reproducibility

Перед релизом проверяется:

```text
compileall
pytest
frozen weights load
real DICOM parse
API /health
API /predict
API /predict-zip
batch contract
quality_prob ∈ [0,1]
exact organizer vocabulary
deterministic repeated inference
hard cases
```

Для study-level оценки соблюдается разделение без смешивания изображений одного исследования между train/validation/test.

## 14. Валидация и ограничения

В доступном обучающем наборе — около 100 исследований и 499 DICOM. Редкие классы (`axis`, `placement`, `ROI`) имеют малое число положительных примеров, поэтому их метрики имеют высокую неопределённость.

В репозитории хранятся engineering/holdout отчёты, но они не являются заменой закрытому тесту организатора и не должны подаваться как независимая клиническая валидация.

## 15. Hard cases

В `reports/hard_cases/` находятся демонстрационные кейсы для:

- axis;
- artifact;
- ROI.

Используйте их для ручной проверки и видеодемонстрации.

## 16. Презентация и материалы

```text
pitch/Танкисты 2.0.pdf
reports/
docs/
```

## 16.1 Публикация в SourceCraft из Windows 10

1. Создайте пустой репозиторий в SourceCraft. Желательно не добавлять туда автоматически созданный README, `.gitignore` или LICENSE: мы уже подготовили их локально.

2. В корне проекта откройте PowerShell:

```powershell
cd C:\DXA_AI

# проверить, что README есть
Get-Item .\README.md

# инициализация
git init
git branch -M main

# вставьте сюда SSH или HTTPS URL из кнопки «Клонировать»
git remote add origin <SOURCECRAFT_REPOSITORY_URL>

# первая отправка
git add .
git status
git commit -m "Final DXA Quality AI submission"
git push -u origin main
```

Если `origin` уже добавлен:

```powershell
git remote set-url origin <SOURCECRAFT_REPOSITORY_URL>
git push -u origin main
```

Есть готовый helper:

```powershell
.\windows\PUSH_TO_SOURCECRAFT_WINDOWS.ps1 -RemoteUrl "<SOURCECRAFT_REPOSITORY_URL>"
```

После push откройте репозиторий в браузере и проверьте, что `README.md` отображается в корне. Если SourceCraft показывает репозиторий как закрытый, включите просмотр для жюри или предоставьте гостевой доступ согласно правилам подачи.

## 16.2 Что отдаём организаторам

**Репозиторий:** URL SourceCraft.

**Документация:** URL на `README.md` или каталог `docs/`.

**Презентация:** ссылка на размещённый PDF `pitch/Танкисты 2.0.pdf` с правом просмотра.

**Прототип:** ссылка на демонстрационное видео; сервис имеет Swagger на `/docs` и endpoints `/health`, `/predict`, `/predict-zip` при локальном запуске.

**Дополнительные материалы:** `reports/`, `docs/`, `pitch/` и demo/hard-case материалы.

## 16.3 Рекомендуемый текст поля «Прототип»

Локальный FastAPI-сервис для автоматизированной проверки качества DEXA-исследований. На вход принимает DICOM или ZIP с исследованиями, выполняет определение анатомической области и оценку качества, возвращает `quality_class`, `quality_prob` и `violation_type` в согласованном словаре организатора. Доступны endpoints `/health`, `/predict` и `/predict-zip`, Swagger/OpenAPI — `/docs`. Сервис контейнеризован и поддерживает пакетную обработку; демонстрация запуска и работы предоставляется в виде видеозаписи.

## 17. Финальный release checklist

Перед отправкой формы:

```text
[ ] README.md находится в корне
[ ] репозиторий открыт для просмотра / guest access
[ ] код и frozen artifacts загружены
[ ] Dockerfile и scripts загружены
[ ] документация загружена
[ ] presentation PDF загружен
[ ] demo/video ссылка подготовлена
[ ] H200 acceptance выполнен на target host
[ ] итоговый CSV batch-проверен
[ ] closed test запускается у организатора
[ ] не опубликованы медицинские датасеты / персональные данные
```

## 18. Соответствие ТЗ

Решение покрывает основной обязательный контур:

- определение анатомической области;
- binary quality classification;
- нарушения укладки/оси/артефактов/ROI;
- structured CSV/JSON output;
- DICOM processing;
- batch processing;
- API;
- контейнеризацию;
- документацию;
- воспроизводимый deployment workflow.

Дополнительные функции вроде DICOM SR и автоматической коррекции разметки могут быть расширены в следующих версиях.

## 19. Контакты команды

Команда: **Танкисты 2.0**

Состав и контактные данные указаны в материалах конкурса/презентации. При публикации репозитория не добавляйте в Git историю медицинские данные, пароли, токены или приватные ключи.
