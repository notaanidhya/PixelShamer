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
$hasVideos = $false

if (Test-Path $MANIFEST) {
    $lines = Get-Content $MANIFEST -ErrorAction SilentlyContinue
    if ($lines -and $lines.Count -gt 1) {
        $hasVideos = $true
    }
}

if (-not $hasVideos) {
    Write-Host "[*] No videos registered in $MANIFEST." -ForegroundColor Yellow
    Write-Host "[*] Running GPU verification test suite (validating CUDA, FP16, and temporal Bi-LSTM)..." -ForegroundColor Yellow
    & $PYTHON ml\deepfake\train_video.py --spatial_checkpoint $SPATIAL_CKPT --epochs 10 --batch_size 32 --num_frames 16 --amp --precache
} else {
    Write-Host "[*] Training on Video Manifest: $MANIFEST ($($lines.Count - 1) video clips)" -ForegroundColor Green
    Write-Host "[*] Leveraging 64GB RAM & GPU via In-Memory Pre-Caching and Multi-Core Workers" -ForegroundColor Cyan
    & $PYTHON ml\deepfake\train_video.py `
        --spatial_checkpoint $SPATIAL_CKPT `
        --manifest_train $MANIFEST `
        --epochs 15 `
        --batch_size 32 `
        --num_frames 16 `
        --num_workers 4 `
        --precache `
        --amp
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "   Phase 2 Execution Complete! Check ml/deepfake/models   " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
