param(
  [Parameter(Mandatory=$true)][string]$DatasetZip
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py=Join-Path $Root '.venv\Scripts\python.exe'
if (!(Test-Path $py)) { throw 'Run windows\1_INSTALL_WINDOWS.bat first.' }
$env:PYTHONPATH=Join-Path $Root 'src'
$out=Join-Path $Root 'reports\windows_visible_test.csv'
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
& $py -m dexa_ai.batch --input $DatasetZip --weights (Join-Path $Root 'artifacts') --out-csv $out --workers 1
Write-Host "Result: $out" -ForegroundColor Green
Import-Csv $out | Format-Table -AutoSize
