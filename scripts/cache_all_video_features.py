"""
scripts/cache_all_video_features.py
===================================
Pre-computes and caches 1408-dim facial feature embeddings for all video clips in data/deepfake/video_manifest.csv.
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import pandas as pd
from ml.deepfake.models.video_model import build_video_model
from ml.deepfake.video_dataset import extract_and_cache_features

def main():
    manifest_csv = ROOT_DIR / "data" / "deepfake" / "video_manifest.csv"
    cache_path = ROOT_DIR / "data" / "deepfake" / "video_features_cache.pt"
    spatial_ckpt = ROOT_DIR / "ml" / "deepfake" / "models" / "efficientnet_deepfake_best.pt"

    print("=" * 65)
    print("   Video Deepfake Feature Cache Generator")
    print("=" * 65)
    df = pd.read_csv(manifest_csv)
    print(f"[*] Manifest: {manifest_csv} ({len(df)} total clips)")
    print(f"[*] Spatial Checkpoint: {spatial_ckpt}")
    print(f"[*] Target Cache: {cache_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Build model to get spatial CNN
    model = build_video_model(
        spatial_checkpoint=str(spatial_ckpt),
        spatial_backbone="efficientnet_b2",
        hidden_dim=256,
        freeze_spatial=True
    ).to(device)

    t0 = time.time()
    extract_and_cache_features(
        manifest_csv=manifest_csv,
        spatial_model=model.spatial_cnn,
        output_cache_path=cache_path,
        device=device,
        num_frames=16,
        batch_size=4
    )
    t1 = time.time()
    print(f"[OK] Caching completed in {t1 - t0:.1f}s ({(t1 - t0)/60:.1f} mins)!")

if __name__ == "__main__":
    main()
