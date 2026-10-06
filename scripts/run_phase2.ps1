# scripts/run_phase2.ps1
# One-click launcher for Phase 2 Video Model Training on Training PC

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Launching Phase 2 Spatio-Temporal Video Deepfake Model " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Resolve Python path
$PYTHON = "python"
if (Test-Path ".\venv\Scripts\python.exe") {
    $PYTHON = ".\venv\Scripts\python.exe"
}

# 2. Check for Phase 1 best spatial checkpoint
$SPATIAL_CKPT = "ml\deepfake\models\efficientnet_b5_deepfake_best.pt"
if (-not (Test-Path $SPATIAL_CKPT)) {
    $SPATIAL_CKPT = "ml\deepfake\models\efficientnet_deepfake_best.pt"
}

Write-Host "[*] Using Spatial Checkpoint: $SPATIAL_CKPT" -ForegroundColor Yellow

# 3. Check for video manifest
$MANIFEST = "data\deepfake\video_manifest.csv"
if (-not (Test-Path $MANIFEST)) {
    Write-Host "[*] No video manifest found at $MANIFEST. Running verification sequence mode..." -ForegroundColor Yellow
    & $PYTHON ml\deepfake\train_video.py --spatial_checkpoint $SPATIAL_CKPT --epochs 10 --batch_size 16 --num_frames 16 --amp
} else {
    Write-Host "[*] Training on Video Manifest: $MANIFEST" -ForegroundColor Green
    & $PYTHON ml\deepfake\train_video.py `
        --spatial_checkpoint $SPATIAL_CKPT `
        --manifest_train $MANIFEST `
        --epochs 15 `
        --batch_size 16 `
        --num_frames 16 `
        --amp
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "   Phase 2 Execution Complete! Check ml/deepfake/models   " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
