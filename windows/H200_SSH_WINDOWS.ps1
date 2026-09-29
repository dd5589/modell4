param(
  [Parameter(Mandatory=$true)][string]$HostName,
  [Parameter(Mandatory=$true)][string]$UserName,
  [Parameter(Mandatory=$true)][string]$ReleaseZip,
  [string]$VisibleTestZip = ''
)
$ErrorActionPreference='Stop'
$release = (Resolve-Path $ReleaseZip).Path
$remoteDir="~/dexa_ai_acceptance"
Write-Host "Creating remote directory $remoteDir" -ForegroundColor Cyan
ssh "$UserName@$HostName" "mkdir -p $remoteDir"
scp $release "$UserName@$HostName`:$remoteDir/release.zip"
if ($VisibleTestZip) {
  $test=(Resolve-Path $VisibleTestZip).Path
  scp $test "$UserName@$HostName`:$remoteDir/visible_test.zip"
}
$cmd=@"
set -e
rm -rf `$HOME/dexa_ai_acceptance/app
mkdir -p `$HOME/dexa_ai_acceptance/app
cd `$HOME/dexa_ai_acceptance
unzip -q -o release.zip -d app
cd app
nvidia-smi -L
python3 -V || true
docker --version
if [ -f scripts/h200_acceptance.sh ]; then
  TESTARG=""
  if [ -f `$HOME/dexa_ai_acceptance/visible_test.zip ]; then TESTARG="`$HOME/dexa_ai_acceptance/visible_test.zip"; fi
  if [ -n "`$TESTARG" ]; then
    bash scripts/h200_acceptance.sh "`$TESTARG" "`$HOME/dexa_ai_acceptance/benchmark" 200
  else
    echo "Release uploaded. Run scripts/h200_acceptance.sh with a test ZIP."
  fi
fi
"@
ssh "$UserName@$HostName" $cmd
