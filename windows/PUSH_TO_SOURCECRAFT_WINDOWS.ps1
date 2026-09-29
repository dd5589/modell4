param(
  [Parameter(Mandatory=$true)]
  [string]$RemoteUrl
)

$ErrorActionPreference = "Stop"

Write-Host "DXA Quality AI -> SourceCraft" -ForegroundColor Cyan
Write-Host "Remote: $RemoteUrl"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "Git for Windows is not installed or git.exe is not in PATH."
}

if (-not (Test-Path "README.md")) {
  throw "Run this script from the repository root (the folder containing README.md)."
}

git init
if (git remote get-url origin 2>$null) {
  git remote set-url origin $RemoteUrl
} else {
  git remote add origin $RemoteUrl
}

git branch -M main
git add .
git status

git commit -m "Final DXA Quality AI submission"
git push -u origin main

Write-Host "SourceCraft push completed." -ForegroundColor Green
