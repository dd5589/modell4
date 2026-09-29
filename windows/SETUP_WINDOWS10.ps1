$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host '=== DXA AI / Windows 10 setup ===' -ForegroundColor Cyan

$pyCmd = $null
try { & py -3.11 --version 2>$null | Out-Null; if ($LASTEXITCODE -eq 0) { $pyCmd = 'py -3.11' } } catch {}
if (-not $pyCmd) {
    try { python --version 2>$null | Out-Null; if ($LASTEXITCODE -eq 0) { $pyCmd = 'python' } } catch {}
}
if (-not $pyCmd) {
    Write-Host 'Python 3.11 not found.' -ForegroundColor Red
    Write-Host 'Install Python 3.11 x64 from python.org, then rerun this script.'
    exit 2
}

Write-Host "Using: $pyCmd"
& cmd /c "$pyCmd --version"

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & cmd /c "$pyCmd -m venv .venv"
}
$Vpy = Join-Path $Root '.venv\Scripts\python.exe'

& $Vpy -m pip install --upgrade pip
& $Vpy -m pip install -r requirements.txt
& $Vpy -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
& $Vpy -m pip install pytest

$env:PYTHONPATH = Join-Path $Root 'src'
& $Vpy -m compileall -q src tests

Write-Host 'Python environment ready.' -ForegroundColor Green
Write-Host 'Run: windows\2_START_API_WINDOWS.bat'
