$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py=Join-Path $Root '.venv\Scripts\python.exe'
if (!(Test-Path $py)) { Write-Host 'Run windows\1_INSTALL_WINDOWS.bat first.' -ForegroundColor Red; exit 2 }
$env:PYTHONPATH=Join-Path $Root 'src'
Write-Host 'Starting DXA AI API at http://127.0.0.1:8000' -ForegroundColor Cyan
Start-Process 'http://127.0.0.1:8000'
& $py -m uvicorn dexa_ai.api:app --host 127.0.0.1 --port 8000
