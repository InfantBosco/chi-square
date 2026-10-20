@echo off
REM Run download script using virtual environment
echo Activating chienv and running download_dataset.py...
call .\chienv\Scripts\activate.bat
python chi_square_project\src\download_dataset.py
pause
