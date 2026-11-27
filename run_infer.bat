@echo off
cd /d "%~dp0"
echo ===================================================
echo Running Neural Network Inference Engine...
echo ===================================================
python infer.py --samples 3
pause
