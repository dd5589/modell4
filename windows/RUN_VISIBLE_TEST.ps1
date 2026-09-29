param([Parameter(Mandatory=$true)][string]$DatasetZip)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py=Join-Path $Root '.venv\Scripts\python.exe'
if (!(Test-Path $py)) { throw 'Run windows\1_INSTALL_WINDOWS.bat first.' }
$env:PYTHONPATH=Join-Path $Root 'src'
$out=Join-Path $Root 'reports\windows_visible_test.csv'
$prepared=Join-Path $Root 'data\visible_test_windows'
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null

$input = Resolve-Path $DatasetZip
$visibleRoot = & $py scripts\extract_visible_test.py --input $input --out-dir $prepared
& $py -m dexa_ai.batch --input $visibleRoot.Trim() --weights (Join-Path $Root 'artifacts') --out-csv $out --workers 1

Write-Host "Result: $out" -ForegroundColor Green
Import-Csv $out | Format-Table -AutoSize
