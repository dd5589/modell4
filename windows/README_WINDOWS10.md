# DXA AI — Windows 10 manual QA workflow

Этот каталог предназначен для работы команды на Windows 10. Нативный Windows Python используется для ручной проверки модели на CPU. Docker на Windows — только для проверки контейнерной упаковки. Финальный H200 acceptance выполняется на выделенной Linux VM с H200; с Windows можно подключиться к ней через OpenSSH.

## Что установить

1. Windows 10 64-bit, желательно 22H2.
2. Python 3.11 x64.
3. Для Docker: Docker Desktop с WSL 2 backend.
4. Для H200: доступ по SSH к VM организатора/команды.

PyTorch 2.10 имеет официальные Windows wheels для CPython 3.11, а официальный способ установки CPU-варианта — через PyTorch CPU index. Docker Desktop поддерживает Windows 10 22H2 с WSL 2; GPU-контейнеры в Windows требуют WSL 2 и NVIDIA GPU на самом Windows-хосте. Для удалённого H200 эти требования относятся к Linux H200 VM, а не к домашнему Windows ПК.

## 1. Установка

Распаковать `dexa_ai_release_final_v2.zip`, открыть каталог проекта и двойным кликом запустить:

`windows\1_INSTALL_WINDOWS.bat`

Скрипт создаст `.venv`, установит фиксированные runtime-зависимости и CPU PyTorch.

## 2. Запуск API

Двойной клик:

`windows\2_START_API_WINDOWS.bat`

Проверить в браузере:

`http://127.0.0.1:8000/health`

Остановить API: Ctrl+C в окне PowerShell.

## 3. Видимый debug test

Двойной клик:

`windows\3_RUN_VISIBLE_TEST_WINDOWS.bat`

Указать путь к `Для теста.zip`. Результат:

`reports\windows_visible_test.csv`

## 4. Самая важная ручная проверка модели

Двойной клик:

`windows\4_RUN_MANUAL_AUDIT_WINDOWS.bat`

Указать **исходный `Датасет.zip`**. Скрипт сам найдёт вложенный `НД_для_обучения.zip`, распакует его, найдёт `разметка.xlsx`, выберет по 20 study для spine и hip (с positive-first sampling) и создаст:

`reports\manual_audit\manual_spine.html`
`reports\manual_audit\manual_hip.html`

Открой эти HTML в браузере и смотри изображение, ground truth и prediction на каждой карточке.

Красная карточка = ручная проверка нужна обязательно.

Особенно проверять:

- spine axis: если `axis_angle_deg <= 5`, модель не должна выводить `Не выравнена ось позвоночника`;
- spine placement: сверху должна быть половина Th12, снизу верхние края подвздошных костей;
- spine artifact: реальное наличие постороннего предмета;
- hip placement/rotation: большой вертел, шейка, седалищная кость и характер малого вертела;
- hip ROI: проверка физических границ по масштабу аппарата;
- exact `violation_type` vocabulary.

## 5. Docker на Windows

Двойной клик:

`windows\5_LOCAL_DOCKER_WINDOWS.bat`

Для этого нужен запущенный Docker Desktop. Этот тест доказывает только, что Linux-контейнер собирается/запускается из Windows; CPU-метрики Windows не являются H200 benchmark.

## 6. H200 с Windows

Если у команды есть удалённая H200 VM, PowerShell умеет выполнять SSH/SCP без установки отдельного Linux-клиента на Windows. Двойной клик:

`windows\6_H200_SSH_WINDOWS.bat`

Он попросит host, SSH user и путь к release ZIP, скопирует пакет на VM и запустит H200 acceptance, если передан visible test ZIP.

Если OpenSSH ещё не доступен, в PowerShell проверь:

`ssh -V`

`scp -V`

## 7. Что прислать после проверки

Для локального manual QA:

`reports\manual_audit\manual_spine.csv`
`reports\manual_audit\manual_hip.csv`
`reports\manual_audit\manual_spine.json`
`reports\manual_audit\manual_hip.json`

Для H200:

`reports/h200_acceptance/health.json`
`reports/h200_acceptance/h200_benchmark.json`
`reports/h200_acceptance/batch_results.csv`
`reports/h200_acceptance/acceptance_summary.json`

## 8. Важное ограничение

Не считать visible `Для теста.zip` closed test организатора. Он предназначен для отладки формата; закрытый набор остаётся на стороне организатора. Метрики закрытого теста считаются только после наличия ground truth у организатора.
