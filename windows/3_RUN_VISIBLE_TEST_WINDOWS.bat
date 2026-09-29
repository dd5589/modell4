@echo off
set /p DATASETZIP=Path to visible test ZIP OR outer Dатасет.zip: 
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_VISIBLE_TEST.ps1" -DatasetZip "%DATASETZIP%"
pause
