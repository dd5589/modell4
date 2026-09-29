@echo off
set /p HOSTNAME=H200 host/IP: 
set /p USERNAME=SSH username: 
set /p RELEASEZIP=Path to dexa_ai_release_final_v2.zip: 
set /p TESTZIP=Optional path to visible test ZIP (press Enter to skip): 
if "%TESTZIP%"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0H200_SSH_WINDOWS.ps1" -HostName "%HOSTNAME%" -UserName "%USERNAME%" -ReleaseZip "%RELEASEZIP%"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0H200_SSH_WINDOWS.ps1" -HostName "%HOSTNAME%" -UserName "%USERNAME%" -ReleaseZip "%RELEASEZIP%" -VisibleTestZip "%TESTZIP%"
)
pause
