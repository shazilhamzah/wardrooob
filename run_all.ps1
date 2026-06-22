# PowerShell script to run both Stage 1 and Stage 2 pipelines in one go

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $PSScriptRoot

Write-Host "========================================" -ForegroundColor Magenta
Write-Host "   Running Wardrooob Unified Pipeline" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

# ── Stage 1: Clothes Measurement ─────────────────────────────────────────────
Write-Host "`n[1/2] Starting Stage 1: Clothes Measurement..." -ForegroundColor Yellow
$Stage1Dir = Join-Path -Path $PSScriptRoot -ChildPath "stage1_clothes_measurement"
$Stage1Script = Join-Path -Path $Stage1Dir -ChildPath "auto\run_garmentiq.ps1"

if (Test-Path $Stage1Script) {
    Write-Host "Executing Stage 1 pipeline..." -ForegroundColor Cyan
    & $Stage1Script
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Stage 1 pipeline completed with warnings or non-zero exit code ($LASTEXITCODE)."
    }
} else {
    Write-Error "Stage 1 script not found at $Stage1Script"
}

# ── Stage 2: Human Modeling ──────────────────────────────────────────────────
Write-Host "`n[2/2] Starting Stage 2: Human Modeling..." -ForegroundColor Yellow
$Stage2Dir = Join-Path -Path $PSScriptRoot -ChildPath "stage2_human_modeling"
$Stage2Script = Join-Path -Path $Stage2Dir -ChildPath "run_pipeline.ps1"

if (Test-Path $Stage2Script) {
    Write-Host "Executing Stage 2 pipeline..." -ForegroundColor Cyan
    & $Stage2Script
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Stage 2 pipeline completed with warnings or non-zero exit code ($LASTEXITCODE)."
    }
} else {
    Write-Error "Stage 2 script not found at $Stage2Script"
}

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "   Unified Pipeline Run Completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Magenta
