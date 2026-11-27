@echo off
cd /d "%~dp0"
echo ===================================================
echo Running Unit Tests & Gradient Verification...
echo ===================================================
python -m unittest discover -s tests -p "test_*.py"
pause
