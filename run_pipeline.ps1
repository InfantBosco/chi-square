# PowerShell pipeline runner script
Write-Host "Activating chienv and running download_dataset.py..." -ForegroundColor Cyan
& ".\chienv\Scripts\python.exe" "chi_square_project\src\download_dataset.py"
