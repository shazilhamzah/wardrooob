# PowerShell script to run the GarmentIQ Accuracy Evaluator

$venvPython = Resolve-Path "$PSScriptRoot\..\venv\Scripts\python.exe" -ErrorAction SilentlyContinue
$scriptPath = Resolve-Path "$PSScriptRoot\..\scripts\evaluate_accuracy.py" -ErrorAction SilentlyContinue

Write-Host "========================================" -ForegroundColor Green
Write-Host "Running GarmentIQ Accuracy Evaluation..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if ($null -eq $venvPython -or -not (Test-Path $venvPython)) {
    Write-Warning "Virtual environment Python not found. Falling back to system Python."
    python $scriptPath
} else {
    & $venvPython $scriptPath
}

if ($LASTEXITCODE -eq 0 -or $?) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Accuracy Evaluation completed!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Error "Accuracy Evaluation failed to execute."
}
