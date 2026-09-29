@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
:menu
cls
echo ================================================
echo        DXA AI - WINDOWS 10 QA MENU
echo ================================================
echo.
echo 1. Install / setup Python environment
necho 2. Start API
necho 3. Run visible test (accepts Datset.zip too)
echo 4. Run manual gold audit (20 spine + 20 hip)
echo 5. Build/run Docker container locally
necho 6. H200 acceptance over SSH
necho 7. Open Windows QA guide
necho 0. Exit
necho.
set /p CHOICE=Choose: 
if "%CHOICE%"=="1" call windows\1_INSTALL_WINDOWS.bat & goto menu
if "%CHOICE%"=="2" call windows\2_START_API_WINDOWS.bat & goto menu
if "%CHOICE%"=="3" call windows\3_RUN_VISIBLE_TEST_WINDOWS.bat & goto menu
if "%CHOICE%"=="4" call windows\4_RUN_MANUAL_AUDIT_WINDOWS.bat & goto menu
if "%CHOICE%"=="5" call windows\5_LOCAL_DOCKER_WINDOWS.bat & goto menu
if "%CHOICE%"=="6" call windows\6_H200_SSH_WINDOWS.bat & goto menu
if "%CHOICE%"=="7" start notepad "%cd%\windows\README_WINDOWS10.md" & goto menu
if "%CHOICE%"=="0" exit /b 0
goto menu
