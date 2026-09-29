@echo off
set /p DATASETZIP=Path to outer Датасет.zip: 
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_MANUAL_AUDIT.ps1" -DatasetZip "%DATASETZIP%" -StudiesPerRegion 20
pause
