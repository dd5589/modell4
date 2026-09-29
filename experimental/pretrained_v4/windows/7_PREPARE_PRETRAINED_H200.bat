@echo off
setlocal
cd /d "%~dp0.."
echo.
echo DXA AI v0.4 - PRETRAINED ENSEMBLE (H200)
echo This launcher prepares the training command. Actual large-backbone training should run on the H200 host.
echo.
set /p HOST=H200 SSH host (example user@10.0.0.10): 
if "%HOST%"=="" goto :end
ssh %HOST% "cd ~/dexa_ai_release && bash scripts/prepare_h200_pretrained.sh /data/train/Isledovaniya /data/train/labels.xlsx artifacts 12"
:end
pause
