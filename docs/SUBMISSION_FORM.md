# Финальная подача — что вставлять в 5 полей

## 1. Репозиторий 🔴

Вставьте URL SourceCraft-репозитория.

Требования перед отправкой:

- репозиторий доступен жюри для просмотра;
- `README.md` лежит в корне;
- исходный код, production weights, Dockerfile и scripts загружены;
- не загружены медицинские DICOM/полный исходный датасет и секреты.

### Push из Windows 10

```powershell
cd C:\DXA_AI
git init
git branch -M main
git remote add origin <SOURCECRAFT_REPOSITORY_URL>
git add .
git status
git commit -m "Final DXA Quality AI submission"
git push -u origin main
```

Или:

```powershell
.\windows\PUSH_TO_SOURCECRAFT_WINDOWS.ps1 -RemoteUrl "<SOURCECRAFT_REPOSITORY_URL>"
```

## 2. Документация 🔴

Основная точка входа:

```text
README.md
```

Подробная документация:

```text
docs/ARCHITECTURE.md
docs/API.md
docs/TRAINING.md
docs/DEPLOYMENT.md
docs/USER_GUIDE.md
docs/FINAL_RELEASE_QA.md
docs/ORGANIZER_CONTRACT.md
docs/H200_RUN.md
docs/TARGET_HOST_AND_CLOSED_TEST.md
```

В форму можно дать ссылку на `README.md` или каталог `docs/`.

## 3. Презентация 🔴

Используйте PDF:

```text
pitch/Танкисты 2.0.pdf
```

Разместите его на разрешённом облаке и включите просмотр по ссылке для жюри.

**Важно:** текущий PDF — существующая версия презентации. После фактического обучения v4, если v4 будет принята в production, синхронизируйте содержательные слайды с фактической архитектурой/метриками. Дизайн менять не требуется.

## 4. Прототип 🔴

Рекомендуемый текст для формы:

> Локальный FastAPI-сервис для автоматизированной проверки качества DEXA-исследований. На вход принимает DICOM или ZIP с исследованиями, выполняет определение анатомической области и оценку качества, возвращает `quality_class`, `quality_prob` и `violation_type` в согласованном словаре организатора. Доступны endpoints `/health`, `/predict` и `/predict-zip`, Swagger/OpenAPI — `/docs`. Сервис контейнеризован и поддерживает пакетную обработку; демонстрация запуска и работы предоставляется в виде видеозаписи.

Если постоянный внешний endpoint не предоставлен, для алгоритмического/backend решения используйте видеодемонстрацию и/или доступный Swagger/test endpoint по правилам организатора.

## 5. Дополнительные материалы 🔴

Рекомендуемый набор:

```text
reports/hard_cases/
reports/submission/
docs/
pitch/
```

Отдельно полезно приложить короткое видео:

```text
Docker/API startup → Swagger → DICOM upload → prediction → CSV batch
```

## Не публиковать

- исходный медицинский датасет;
- DICOM с медицинскими изображениями;
- пароли, API keys, SSH private keys;
- локальные служебные пути;
- hidden/closed test организатора.
