"""
scripts/train_faceforensics_c23.py
====================================
Automated Training Pipeline for FaceForensics++ (c23, ~18-40 GB) on Training PC.

Pipeline:
1. Scans data/FaceForensics++_C23 for original YouTube videos and 6 manipulation categories
   (Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures, DeepFakeDetection).
2. Builds a balanced multi-manipulation manifest (e.g. 1,000 Real + 1,000 balanced Fake).
3. Pre-computes 2048-dim EfficientNet-B5 features into RAM/disk cache (~262 MB).
4. Trains Regularized Spatio-Temporal Model with Projection Bottleneck.
5. Saves lightweight checkpoint to 'ml/deepfake/models/deepfake_video_ff_c23.pt' (~3.8 MB).
"""

import os
import sys
import time
import argparse
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

def build_faceforensics_manifest(source_dir: Path, output_csv: Path, max_samples: int = 2000) -> pd.DataFrame:
    """Builds a balanced manifest across all FaceForensics++ manipulation categories."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    video_exts = ("*.mp4", "*.avi", "*.mov")
    
    print(f"[*] Scanning FaceForensics++ directory for video files...")
    print(f"    Target directory: {source_dir.resolve()}")
    all_videos = []
    for ext in video_exts:
        all_videos.extend(list(source_dir.rglob(ext)))

    real_records = []
    fake_by_category = {
        "deepfakes": [],
        "face2face": [],
        "faceswap": [],
        "faceshifter": [],
        "neuraltextures": [],
        "deepfakedetection": [],
        "other": []
    }

    for p in all_videos:
        p_str = str(p).lower().replace("\\", "/")
        parts = [part.lower() for part in p.parts]
        
        # Skip non-video helper directories
        if "csv" in parts:
            continue

        # Real candidates: original, youtube, real, original_sequences
        if any(part in ["original", "youtube", "real", "original_sequences"] for part in parts) or "/original/" in p_str:
            real_records.append({"filepath": str(p.resolve()), "filename": p.name, "label": 0, "category": "real"})
        elif any("faceshifter" in part for part in parts):
            fake_by_category["faceshifter"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "faceshifter"})
        elif any("deepfakedetection" in part or "dfd" in part for part in parts):
            fake_by_category["deepfakedetection"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "deepfakedetection"})
        elif any("deepfake" in part for part in parts):
            fake_by_category["deepfakes"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "deepfakes"})
        elif any("face2face" in part for part in parts):
            fake_by_category["face2face"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "face2face"})
        elif any("faceswap" in part for part in parts):
            fake_by_category["faceswap"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "faceswap"})
        elif any("neuraltextures" in part for part in parts):
            fake_by_category["neuraltextures"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "neuraltextures"})
        else:
            fake_by_category["other"].append({"filepath": str(p.resolve()), "filename": p.name, "label": 1, "category": "manipulated"})

    total_fakes = sum(len(v) for v in fake_by_category.values())
    print(f"[*] Discovered {len(real_records)} Real videos and {total_fakes} Fake videos across categories:")
    for cat, items in fake_by_category.items():
        if len(items) > 0:
            print(f"    - {cat.capitalize()}: {len(items)} clips")

    active_fake_categories = {k: v for k, v in fake_by_category.items() if len(v) > 0}
    num_fake_cats = len(active_fake_categories)
    
    if len(real_records) == 0:
        raise RuntimeError("No real/original videos found. Please ensure 'original' directory exists.")
    if num_fake_cats == 0:
        raise RuntimeError("No manipulated videos found in any category.")

    # Determine balanced sampling counts
    if max_samples and max_samples > 0:
        target_real_count = min(len(real_records), max_samples // 2)
    else:
        target_real_count = len(real_records)

    per_cat_quota = max(1, target_real_count // num_fake_cats)
    
    selected_fakes = []
    for cat, items in active_fake_categories.items():
        np.random.seed(42)
        shuffled = list(items)
        np.random.shuffle(shuffled)
        selected_fakes.extend(shuffled[:per_cat_quota])

    np.random.seed(42)
    shuffled_reals = list(real_records)
    np.random.shuffle(shuffled_reals)
    selected_reals = shuffled_reals[:min(len(shuffled_reals), len(selected_fakes))]
    
    # Ensure exact 1:1 balance
    selected_fakes = selected_fakes[:len(selected_reals)]

    all_records = selected_reals + selected_fakes
    np.random.seed(42)
    np.random.shuffle(all_records)
    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)
    
    print(f"\n[OK] Created balanced FaceForensics++ manifest at: {output_csv}")
    print(f"     Total clips: {len(df)} ({len(selected_reals)} Real, {len(selected_fakes)} Fake)")
    return df

def find_dataset_root(cli_dir: str | None = None) -> Path | None:
    """Finds FaceForensics++ folder among standard directories."""
    possible_roots = []
    if cli_dir:
        possible_roots.append(Path(cli_dir))

    possible_roots.extend([
        BASE_DIR / "data" / "FaceForensics++_C23",
        BASE_DIR / "data" / "faceforensics",
        BASE_DIR / "data" / "faceforencis",
        Path("C:/Users/vitbopal/Downloads/deepfake project/PixelShamer/data/FaceForensics++_C23"),
        Path.home() / "Downloads" / "deepfake project" / "PixelShamer" / "data" / "FaceForensics++_C23",
        Path.home() / "Downloads" / "FaceForensics++_C23",
        Path.home() / "Downloads" / "faceforensics",
        Path.home() / "Downloads" / "faceforencis",
        Path("C:/Users/vitbopal/Downloads/faceforensics"),
        Path("C:/Users/vitbopal/Downloads/faceforencis"),
        Path.home() / "Downloads",
        BASE_DIR / "data",
    ])

    for root in possible_roots:
        if root.exists() and root.is_dir():
            # Quick check if directory contains any .mp4 files without slow exhaustive scan
            first_mp4 = next(root.rglob("*.mp4"), None)
            if first_mp4 is not None:
                return root
    return None

def main():
    parser = argparse.ArgumentParser(description="FaceForensics++ c23 Spatio-Temporal Training Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to FaceForensics++ dataset directory")
    parser.add_argument("--max-samples", type=int, default=2000, help="Max total samples (e.g. 2000 = 1000 real + 1000 fake)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--force-recompute", action="store_true", help="Force feature recomputation")
    args = parser.parse_args()

    print("=" * 75)
    print("   FACEFORENSICS++ (c23) REGULARIZED SPATIO-TEMPORAL TRAINING")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Locate dataset directory
    data_root = find_dataset_root(args.data_dir)
    if data_root is None:
        print("[!] ERROR: Could not locate FaceForensics++ videos (.mp4 files).")
        print("    Please provide --data-dir path or place extracted dataset into data/FaceForensics++_C23.")
        return

    print(f"[*] Found FaceForensics++ dataset at: {data_root.resolve()}")

    # Build manifest
    manifest_csv = BASE_DIR / "data" / "deepfake" / "ff_c23_manifest.csv"
    df = build_faceforensics_manifest(data_root, manifest_csv, max_samples=args.max_samples)

    # Paths
    spatial_ckpt = BASE_DIR / "ml" / "deepfake" / "models" / "efficientnet_b5_deepfake_best.pt"
    feature_cache_path = BASE_DIR / "data" / "deepfake" / "ff_c23_features.pt"
    output_ckpt = BASE_DIR / "ml" / "deepfake" / "models" / "deepfake_video_ff_c23.pt"

    if not spatial_ckpt.exists():
        print(f"[!] ERROR: Spatial backbone checkpoint not found at {spatial_ckpt}")
        return

    recompute = args.force_recompute
    if not recompute and feature_cache_path.exists():
        try:
            chk = torch.load(feature_cache_path, map_location="cpu", weights_only=True)
            if len(chk.get("features", [])) == len(df):
                print(f"[*] Existing feature cache valid ({len(df)} clips). Skipping extraction.")
            else:
                print(f"[*] Cache size mismatch ({len(chk.get('features', []))} vs {len(df)}). Recomputing...")
                recompute = True
        except Exception:
            recompute = True
    else:
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
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
    print(f"          Size: {output_ckpt.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 75)

if __name__ == "__main__":
    main()
