$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $RootDir

$PythonPath = Join-Path -Path $RootDir -ChildPath "venv\Scripts\python.exe"

if (-Not (Test-Path $PythonPath)) {
    Write-Host "Python virtual environment not found at $PythonPath!" -ForegroundColor Red
    exit 1
}

Write-Host "Starting SHAPY Pipeline..." -ForegroundColor Green

# 0. Extract OpenPose Keypoints
Write-Host "`n--- Step 0: Extracting Keypoints with MediaPipe ---" -ForegroundColor Cyan
& $PythonPath extract_keypoints.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Keypoint Extraction failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 1. Run Shape Regressor
Write-Host "`n--- Step 1: Running Shape Regressor (Images to 3D Meshes & Parameters) ---" -ForegroundColor Cyan
Set-Location -Path "regressor"
& $PythonPath demo.py --save-vis true --save-params true --save-mesh true --split test --datasets openpose --output-folder ../samples/shapy_fit/ --exp-cfg configs/b2a_expose_hrnet_demo.yaml --exp-opts output_folder=../data/trained_models/shapy/SHAPY_A part_key=pose datasets.pose.openpose.data_folder=../samples datasets.pose.openpose.img_folder=images datasets.pose.openpose.keyp_folder=openpose datasets.batch_size=1 datasets.pose_shape_ratio=1.0
if ($LASTEXITCODE -ne 0) {
    Write-Host "Shape Regressor failed with exit code $LASTEXITCODE" -ForegroundColor Red
    Set-Location -Path ".."
    exit $LASTEXITCODE
}
Set-Location -Path ".."

# 2. Run Virtual Measurements
Write-Host "`n--- Step 2: Running Virtual Measurements (3D Parameters to Anthropometric Measurements) ---" -ForegroundColor Cyan
Set-Location -Path "measurements"
# Use the output from the previous step as input for measurements
& $PythonPath virtual_measurements.py --input-folder ../samples/shapy_fit/ --output-folder ../samples/virtual_measurements/
if ($LASTEXITCODE -ne 0) {
    Write-Host "Virtual Measurements failed with exit code $LASTEXITCODE" -ForegroundColor Red
    Set-Location -Path ".."
    exit $LASTEXITCODE
}
Set-Location -Path ".."

Write-Host "`nPipeline completed successfully! Outputs are in the 'samples/shapy_fit' and 'samples/virtual_measurements' folders." -ForegroundColor Green
