@echo off
title The Retail Intelligence Ledger
echo Activating virtual environment...
call .\chienv\Scripts\activate.bat
echo Starting application...
python run_app.py
pause
