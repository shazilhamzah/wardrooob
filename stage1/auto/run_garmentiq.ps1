# PowerShell script to run the unified GarmentIQ pipeline

$venvPython = Resolve-Path "$PSScriptRoot\..\venv\Scripts\python.exe" -ErrorAction SilentlyContinue
$scriptPath = Resolve-Path "$PSScriptRoot\..\scripts\run_garmentiq.py" -ErrorAction SilentlyContinue

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running GarmentIQ Unified Pipeline..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($null -eq $venvPython -or -not (Test-Path $venvPython)) {
    Write-Warning "Virtual environment Python not found. Falling back to system Python."
    python $scriptPath
} else {
    & $venvPython $scriptPath
}

if ($LASTEXITCODE -eq 0 -or $?) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "GarmentIQ Pipeline completed!" -ForegroundColor Green
    Write-Host "Check the 'output' directory for results." -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
} else {
    Write-Error "Pipeline failed to execute properly."
}
