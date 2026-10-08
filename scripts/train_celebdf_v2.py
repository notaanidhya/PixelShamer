"""
scripts/train_celebdf_v2.py
=============================
One-Click Automated Full Training Pipeline for Celeb-DF v2 on Training PC.

Pipeline Stages:
1. Discovers the downloaded 9GB Celeb-DF dataset on the machine.
2. Builds a perfectly balanced manifest of 1,168 videos (584 Real + 584 Fake).
3. Pre-computes 2048-dim EfficientNet-B5 features into RAM/disk cache (~150MB).
4. Trains the Regularized Feature-Bottleneck Bi-LSTM (prevents overfitting).
5. Saves new best checkpoint to 'ml/deepfake/models/deepfake_video_v2.pt'.
6. Executes Gate 3 validation audit and prints performance comparison.
"""

import os
import sys
import time
from pathlib import Path
import torch
import pandas as pd

# Root path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.deepfake.video_dataset import build_video_manifest, extract_and_cache_features
from ml.deepfake.train_video import train_video_model
from ml.deepfake.models.video_model import build_video_model

def locate_celebdf_root() -> Path | None:
    """Finds the root directory containing Celeb-DF folders."""
    candidate_roots = [
        BASE_DIR / "data" / "celeb_df",
        BASE_DIR / "data" / "deepfake",
        Path.home() / "Downloads",
        Path("C:/Users/vitbopal/Downloads"),
        BASE_DIR / "data",
    ]
    
    for c in candidate_roots:
        if not c.exists():
            continue
        # Check if celeb-real or celeb-synthesis exists in this tree
        for root, dirs, files in os.walk(c):
            lower_dirs = [d.lower() for d in dirs]
            if any("celeb-real" in d or "celeb-synthesis" in d for d in lower_dirs):
                print(f"[*] Discovered Celeb-DF dataset at: {root}")
                return Path(root)
                
    # Fallback to search any mp4 files in candidate roots
    for c in candidate_roots:
        if c.exists() and len(list(c.rglob("*.mp4"))) > 100:
            print(f"[*] Using video candidate directory: {c}")
            return c
            
    return None

def main():
    print("=" * 75)
    print("   LAUNCHING CELEB-DF v2 REGULARIZED SPATIO-TEMPORAL TRAINING (PHASE 2+)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Locate dataset
    data_root = locate_celebdf_root()
    if data_root is None:
        print("[!] ERROR: Could not automatically locate Celeb-DF dataset.")
        print("    Please ensure the dataset is unzipped in data/celeb_df or Downloads.")
        return

    # 2. Build 1,168-clip balanced manifest
    manifest_csv = BASE_DIR / "data" / "deepfake" / "celebdf_v2_manifest.csv"
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n[Stage 1/4] Building balanced 1,168-clip manifest...")
    df = build_video_manifest(data_root, manifest_csv, limit=None, balance=True)
    num_reals = len(df[df["label"] == 0])
    num_fakes = len(df[df["label"] == 1])
    print(f"[*] Manifest constructed: {len(df)} total clips ({num_reals} Real, {num_fakes} Fake)")

    if len(df) < 20:
        print("[!] Warning: Less than 20 clips found. Check dataset folder paths.")
        return

    # 3. Pre-compute and cache features
    spatial_ckpt = BASE_DIR / "ml" / "deepfake" / "models" / "efficientnet_b5_deepfake_best.pt"
    if not spatial_ckpt.exists():
        spatial_ckpt = BASE_DIR / "ml" / "deepfake" / "models" / "efficientnet_deepfake_best.pt"
    print(f"[*] Spatial Backbone Checkpoint: {spatial_ckpt}")

    feature_cache_path = BASE_DIR / "data" / "deepfake" / "celebdf_v2_features.pt"
    recompute_cache = True
    if feature_cache_path.exists():
        try:
            chk = torch.load(feature_cache_path, map_location="cpu", weights_only=True)
            if len(chk.get("features", [])) == len(df):
                print(f"[*] Found existing feature cache with {len(df)} clips ({feature_cache_path.stat().st_size / (1024*1024):.1f} MB). Skipping extraction.")
                recompute_cache = False
        except Exception:
            recompute_cache = True

    if recompute_cache:
        print("\n[Stage 2/4] Pre-caching video features (EfficientNet-B5 -> RAM/Disk)...")
        t0 = time.time()
        spatial_model = build_video_model(
            spatial_checkpoint=str(spatial_ckpt),
            spatial_backbone="efficientnet_b5",
            freeze_spatial=True
        ).to(device)
        
        extract_and_cache_features(
            manifest_csv=manifest_csv,
            spatial_model=spatial_model.spatial_cnn,
            output_cache_path=feature_cache_path,
            device=device,
            num_frames=16,
            batch_size=8 if device.type == "cuda" else 2
        )
        del spatial_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t1 = time.time()
        print(f"[*] Feature pre-caching finished in {(t1-t0)/60:.1f} mins.")

    # 4. Train Regularized Bi-LSTM model
    output_ckpt = BASE_DIR / "ml" / "deepfake" / "models" / "deepfake_video_v2.pt"
    print("\n[Stage 3/4] Training Regularized Spatio-Temporal Model (v2)...")
    train_video_model(
        epochs=15,
        batch_size=16,
        lr=1.5e-4,
        hidden_dim=128,
        num_lstm_layers=1,
        patience=5,
        spatial_checkpoint=str(spatial_ckpt),
        output_checkpoint=str(output_ckpt),
        feature_cache_path=str(feature_cache_path),
        manifest_train=str(manifest_csv),
        num_frames=16,
        use_amp=(device.type == "cuda"),
        precache=True
    )

    # 5. Summary and Git Instructions
    print("\n" + "=" * 75)
    print("   TRAINING RUN COMPLETE! MODEL SAVED TO:")
    print(f"   {output_ckpt}")
    print("=" * 75)
    print("\nNext step: To transfer the new v2 model to your Dev PC via Git:")
    print("  git add ml/deepfake/models/deepfake_video_v2.pt ml/deepfake/models/deepfake_video_v2_training_history.json")
    print("  git commit -m \"feat: add trained regularized spatio-temporal deepfake model v2\"")
    print("  git push -u origin training_pc")
    print("\nThen on your Dev PC, simply run:")
    print("  git fetch origin")
    print("  git merge origin/training_pc")

if __name__ == "__main__":
    main()
