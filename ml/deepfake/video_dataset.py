"""
ml/deepfake/video_dataset.py
============================
PyTorch Datasets & Feature Caching for Deepfake Video Sequences.
Supports:
  1. On-the-fly video decoding via VideoFaceExtractor
  2. In-memory / Pre-cached feature sequence dataset (optimizing 64GB RAM)
  3. Video directory scanner and manifest builder for DFDC / FaceForensics++
"""

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ml.deepfake.video_extractor import VideoFaceExtractor

class VideoForensicsDataset(Dataset):
    """
    On-the-fly video sequence dataset. Decodes video and extracts 16 face crops per sample.
    """
    def __init__(
        self,
        manifest_csv: str | Path,
        num_frames: int = 16,
        target_size: int = 288,
        is_training: bool = True
    ):
        self.df = pd.read_csv(manifest_csv)
        self.num_frames = num_frames
        self.is_training = is_training
        self.extractor = VideoFaceExtractor(target_size=target_size, default_num_frames=num_frames)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = row["filepath"]
        label = float(row["label"])

        try:
            tensor_seq, _ = self.extractor.extract_face_sequence(
                video_path=video_path,
                num_frames=self.num_frames,
                return_tensors=True
            )
        except Exception as e:
            tensor_seq = None

        # Fallback to zero tensor if video read failed
        if tensor_seq is None or not isinstance(tensor_seq, torch.Tensor):
            tensor_seq = torch.zeros((self.num_frames, 3, self.extractor.target_size, self.extractor.target_size), dtype=torch.float32)

        label_tensor = torch.tensor([label], dtype=torch.float32)
        return tensor_seq, label_tensor, Path(video_path).name

class CachedFeatureDataset(Dataset):
    """
    High-performance feature dataset stored directly in RAM.
    Bypasses OpenCV decoding during temporal training.
    """
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, names: list[str] | None = None):
        self.features = features.float() # (N, T, 2048)
        self.labels = labels.float()     # (N, 1)
        self.names = names or [f"sample_{i}" for i in range(len(features))]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx], self.names[idx]

def extract_and_cache_features(
    manifest_csv: str | Path,
    spatial_model: torch.nn.Module,
    output_cache_path: str | Path,
    device: torch.device,
    num_frames: int = 16,
    batch_size: int = 4
) -> Path:
    """
    Precomputes and caches (N, 16, 2048) feature representations to disk/RAM.
    Leverages 64GB RAM to load thousands of clips into memory in seconds.
    """
    output_cache_path = Path(output_cache_path)
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = VideoForensicsDataset(manifest_csv, num_frames=num_frames, is_training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    spatial_model.eval()
    all_features = []
    all_labels = []
    all_names = []

    print(f"[*] Pre-computing features for {len(dataset)} videos onto {device}...")

    with torch.no_grad():
        for i, (video_tensors, labels, names) in enumerate(loader):
            # video_tensors: (B, T, 3, H, W)
            b, t, c, h, w = video_tensors.shape
            x_flat = video_tensors.view(b * t, c, h, w).to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                feats = spatial_model.backbone.forward_features(x_flat)
                pooled = spatial_model.backbone.forward_head(feats, pre_logits=True)

            pooled_seq = pooled.view(b, t, -1).cpu() # (B, T, 2048)
            all_features.append(pooled_seq)
            all_labels.append(labels)
            all_names.extend(names)

            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                print(f"    - Processed {(i + 1) * batch_size}/{len(dataset)} videos...")

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    cache_dict = {
        "features": features_tensor,
        "labels": labels_tensor,
        "names": all_names,
        "num_frames": num_frames,
        "feature_dim": features_tensor.shape[-1]
    }

    torch.save(cache_dict, output_cache_path)
    size_mb = output_cache_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Feature cache saved to {output_cache_path} ({size_mb:.1f} MB, {len(features_tensor)} samples)")

    return output_cache_path

def build_video_manifest(source_dir: str | Path, output_csv: str | Path) -> pd.DataFrame:
    """
    Builds a CSV manifest from directories containing video files:
    Looks for (real / fake) folders inside source_dir.
    """
    source_path = Path(source_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")

    real_candidates = [source_path / "real", source_path / "original", source_path / "actors"]
    fake_candidates = [source_path / "fake", source_path / "manipulated", source_path / "deepfake"]

    real_dir = next((d for d in real_candidates if d.exists()), None)
    fake_dir = next((d for d in fake_candidates if d.exists()), None)

    if real_dir is None or fake_dir is None:
        # Fallback: scan all videos and infer label from folder name
        for ext in video_exts:
            for p in source_path.rglob(ext):
                parent_name = p.parent.name.lower()
                is_fake = "fake" in parent_name or "manipulated" in parent_name
                records.append({
                    "filepath": str(p.resolve()),
                    "filename": p.name,
                    "label": 1 if is_fake else 0,
                    "label_name": "fake" if is_fake else "real"
                })
    else:
        for ext in video_exts:
            for p in real_dir.glob(ext):
                records.append({
                    "filepath": str(p.resolve()),
                    "filename": p.name,
                    "label": 0,
                    "label_name": "real"
                })
            for p in fake_dir.glob(ext):
                records.append({
                    "filepath": str(p.resolve()),
                    "filename": p.name,
                    "label": 1,
                    "label_name": "fake"
                })

    if not records:
        print(f"[!] NOTICE: No video files (.mp4/.avi/.mov) found in: {source_path}")
        print("    Please place video files into subfolders like: data/video_clips/real and data/video_clips/fake")
        df = pd.DataFrame(columns=["filepath", "filename", "label", "label_name"])
    else:
        df = pd.DataFrame(records)

    df.to_csv(output_path, index=False)
    print(f"[OK] Created video manifest at {output_path} ({len(df)} videos found)")
    return df
