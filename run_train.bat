@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting Deep Learning Training Pipeline...
echo ===================================================
python train.py
pause
