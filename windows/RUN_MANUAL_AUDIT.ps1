param(
  [Parameter(Mandatory=$true)][string]$DatasetZip,
  [int]$StudiesPerRegion = 20
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py=Join-Path $Root '.venv\Scripts\python.exe'
if (!(Test-Path $py)) { throw 'Run windows\1_INSTALL_WINDOWS.bat first.' }
$env:PYTHONPATH=Join-Path $Root 'src'
$prepared=Join-Path $Root 'data\windows_prepared'

& $py scripts\prepare_dataset.py --training-zip $DatasetZip --out-dir $prepared
$trainRoot=Join-Path $prepared 'training_root'
$labels=(Get-ChildItem -Path $trainRoot -Filter *.xlsx -Recurse | Select-Object -First 1).FullName
if (-not $labels) { throw 'Could not find annotation XLSX after preparation.' }

$outDir=Join-Path $Root 'reports\manual_audit'
New-Item -ItemType Directory -Force $outDir | Out-Null

foreach ($region in @('spine','hip')) {
  $csv=Join-Path $outDir "manual_${region}.csv"
  $html=Join-Path $outDir "manual_${region}.html"
  $json=Join-Path $outDir "manual_${region}.json"
  & $py scripts\manual_validation.py `
    --data-root $trainRoot `
    --labels-xlsx $labels `
    --weights (Join-Path $Root 'artifacts') `
    --region $region `
    --n $StudiesPerRegion `
    --seed 42 `
    --out-csv $csv `
    --out-html $html `
    --out-json $json
  Start-Process $html
}
Write-Host 'Manual audit finished. Review the two HTML reports.' -ForegroundColor Green
