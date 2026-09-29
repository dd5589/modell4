$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI not found. Install Docker Desktop with WSL2 backend.' }
docker info | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop is not running.' }
Write-Host 'Building Linux container...' -ForegroundColor Cyan
docker build -t dexa-ai:windows-check .
Write-Host 'Starting container on http://127.0.0.1:8000' -ForegroundColor Green
docker run --rm -p 8000:8000 dexa-ai:windows-check
