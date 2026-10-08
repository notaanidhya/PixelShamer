"""
scripts/train_faceforensics_c23.py
====================================
Automated Training Pipeline for FaceForensics++ (c23, ~35-40 GB) on Training PC.

Pipeline:
1. Scans data/faceforensics for original YouTube videos and 4 manipulation categories
   (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
2. Builds a balanced multi-manipulation manifest (1,000 Real + 1,000 balanced Fake).
3. Pre-computes 2048-dim EfficientNet-B5 features into RAM/disk cache (~262 MB).
4. Trains Regularized Spatio-Temporal Model with Projection Bottleneck.
5. Saves lightweight checkpoint to 'ml/deepfake/models/deepfake_video_ff_c23.pt' (3.8 MB).
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.deepfake.video_dataset import extract_and_cache_features
from ml.deepfake.train_video import train_video_model
from ml.deepfake.models.video_model import build_video_model

def build_faceforensics_manifest(source_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Builds a balanced manifest across all FaceForensics++ manipulation categories."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    video_exts = ("*.mp4", "*.avi", "*.mov")
    
    print("[*] Scanning FaceForensics++ directory for video files...")
    all_videos = []
    for ext in video_exts:
        all_videos.extend(list(source_dir.rglob(ext)))

    real_records = []
    fake_by_category = {
        "deepfakes": [],
        "face2face": [],
        "faceswap": [],
        "neuraltextures": [],
        "other": []
    }

    for p in all_videos:
        p_str = str(p).lower().replace("\\", "/")
        parent = p.parent.name.lower()
        
        # Real candidates
        if "original_sequences" in p_str or "youtube" in p_str or parent in ["real", "original"]:
            real_records.append({"filepath": str(p.resolve()), "filename": p.name, "label": 0, "category": "real"})
        elif "deepfake" in p_str:
            fake_by_category["deepfakes"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "deepfakes"})
        elif "face2face" in p_str:
            fake_by_category["face2face"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "face2face"})
        elif "faceswap" in p_str:
            fake_by_category["faceswap"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "faceswap"})
        elif "neuraltextures" in p_str:
            fake_by_category["neuraltextures"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "neuraltextures"})
        elif "manipulated" in p_str:
            fake_by_category["other"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "manipulated"})

    total_fakes = sum(len(v) for v in fake_by_category.values())
    print(f"[*] Discovered {len(real_records)} Real videos and {total_fakes} Fake videos across categories:")
    for cat, items in fake_by_category.items():
        if len(items) > 0:
            print(f"    - {cat.capitalize()}: {len(items)} clips")

    # Sample evenly across manipulation types to match total real records
    target_real_count = min(len(real_records), 1000)
    per_cat_quota = max(1, target_real_count // max(1, len([k for k, v in fake_by_category.items() if len(v) > 0])))
    
    selected_fakes = []
    for cat, items in fake_by_category.items():
        if len(items) > 0:
            np.random.seed(42)
            shuffled = list(items)
            np.random.shuffle(shuffled)
            selected_fakes.extend(shuffled[:per_cat_quota])

    np.random.seed(42)
    shuffled_reals = list(real_records)
    np.random.shuffle(shuffled_reals)
    selected_reals = shuffled_reals[:len(selected_fakes)]

    all_records = selected_reals + selected_fakes
    np.random.shuffle(all_records)
    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)
    
    print(f"\n[OK] Created balanced FaceForensics++ manifest at {output_csv}")
    print(f"     Total clips: {len(df)} ({len(selected_reals)} Real, {len(selected_fakes)} Fake)")
    return df

def main():
    print("=" * 75)
    print("   FACEFORENSICS++ (c23) REGULARIZED TRAINING PIPELINE")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Locate directory
    possible_roots = [
        BASE_DIR / "data" / "faceforensics",
        BASE_DIR / "data" / "faceforencis",
        Path.home() / "Downloads" / "faceforensics",
        Path.home() / "Downloads" / "faceforencis",
        Path("C:/Users/vitbopal/Downloads/faceforensics"),
        Path("C:/Users/vitbopal/Downloads/faceforencis"),
        Path.home() / "Downloads",
        BASE_DIR / "data",
    ]
    
    data_root = None
    for root in possible_roots:
        if root.exists() and len(list(root.rglob("*.mp4"))) > 100:
            data_root = root
            print(f"[*] Found FaceForensics++ dataset at: {data_root}")
            break

    if data_root is None:
        print("[!] ERROR: Could not locate FaceForensics++ videos (.mp4 files).")
        print("    Please run 'python scripts/download_faceforensics_direct.py' first.")
        return

    # Build manifest
    manifest_csv = BASE_DIR / "data" / "deepfake" / "ff_c23_manifest.csv"
    df = build_faceforensics_manifest(data_root, manifest_csv)

    # Pre-cache features
    spatial_ckpt = BASE_DIR / "ml" / "deepfake" / "models" / "efficientnet_b5_deepfake_best.pt"
    feature_cache_path = BASE_DIR / "data" / "deepfake" / "ff_c23_features.pt"
    output_ckpt = BASE_DIR / "ml" / "deepfake" / "models" / "deepfake_video_ff_c23.pt"

    recompute = True
    if feature_cache_path.exists():
        try:
            chk = torch.load(feature_cache_path, map_location="cpu", weights_only=True)
            if len(chk.get("features", [])) == len(df):
                print(f"[*] Existing feature cache valid ({len(df)} clips). Skipping extraction.")
                recompute = False
        except Exception:
            recompute = True

    if recompute:
        print("\n[Stage 1/2] Pre-caching video features to RAM/disk (2048-dim embeddings)...")
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
        print(f"[*] Pre-caching completed in {(t1-t0)/60:.1f} mins.")

    # Train model
    print("\n[Stage 2/2] Training Spatio-Temporal Model on FaceForensics++...")
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

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Training complete! Model saved to: {output_ckpt}")
    print("=" * 75)

if __name__ == "__main__":
    main()
