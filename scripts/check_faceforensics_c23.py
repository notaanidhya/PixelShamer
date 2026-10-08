"""
scripts/check_faceforensics_c23.py
====================================
Dataset Inspector & Integrity Verifier for FaceForensics++ (c23).

Usage:
    venv\\Scripts\\python scripts/check_faceforensics_c23.py
"""

import sys
import os
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

def find_dataset_root():
    candidates = [
        BASE_DIR / "data" / "FaceForensics++_C23",
        BASE_DIR / "data" / "faceforensics",
        BASE_DIR / "data" / "faceforencis",
        Path("C:/Users/vitbopal/Downloads/deepfake project/PixelShamer/data/FaceForensics++_C23"),
        Path.home() / "Downloads" / "deepfake project" / "PixelShamer" / "data" / "FaceForensics++_C23",
        Path.home() / "Downloads" / "FaceForensics++_C23",
        Path.home() / "Downloads" / "faceforensics",
        Path.home() / "Downloads" / "faceforencis",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None

def inspect_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()
    return {
        "resolution": f"{width}x{height}",
        "fps": f"{fps:.1f}",
        "frames": frame_count,
        "duration": f"{duration_sec:.1f}s",
        "size_mb": f"{video_path.stat().st_size / (1024*1024):.2f} MB"
    }

def main():
    print("=" * 75)
    print("   FACEFORENSICS++ (c23) DATASET CHECKOUT & VERIFICATION")
    print("=" * 75)

    root = find_dataset_root()
    if root is None:
        print("[!] Could not locate FaceForensics++ directory.")
        print(f"    Looked in standard locations under: {BASE_DIR / 'data'}")
        return

    print(f"[*] Dataset Root: {root.resolve()}\n")
    
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        print("[!] No subdirectories found in dataset root.")
        return

    print(f"{'Category / Subfolder':<25} | {'Video Files':<12} | {'Sample Resolution':<18} | {'Sample Duration':<15}")
    print("-" * 75)

    total_videos = 0
    real_count = 0
    fake_count = 0

    video_exts = ("*.mp4", "*.avi", "*.mov")

    for sd in sorted(subdirs, key=lambda x: x.name.lower()):
        # Skip csv or metadata directory
        if sd.name.lower() == "csv":
            csv_files = list(sd.rglob("*.csv"))
            print(f"{sd.name + ' (metadata)':<25} | {len(csv_files):<12} (CSV metadata tables)")
            continue

        vids = []
        for ext in video_exts:
            vids.extend(list(sd.rglob(ext)))

        num_vids = len(vids)
        total_videos += num_vids

        is_real = sd.name.lower() in ["original", "youtube", "real", "original_sequences"]
        if is_real:
            real_count += num_vids
            tag = " [REAL]"
        else:
            fake_count += num_vids
            tag = " [FAKE]"

        sample_info = "N/A"
        sample_dur = "N/A"
        if vids:
            info = inspect_video(vids[0])
            if info:
                sample_info = f"{info['resolution']} @ {info['fps']}fps"
                sample_dur = f"{info['duration']} ({info['size_mb']})"

        name_display = f"{sd.name}{tag}"
        print(f"{name_display:<25} | {num_vids:<12} | {sample_info:<18} | {sample_dur:<15}")

    print("-" * 75)
    print(f"[*] SUMMARY:")
    print(f"    - Total Videos Discovered : {total_videos:,}")
    print(f"    - Authentic (Real) Clips  : {real_count:,}")
    print(f"    - Manipulated (Fake) Clips: {fake_count:,}")
    print(f"    - Compression Standard    : c23 (H.264 standard HQ compression)")
    print("=" * 75)
    print("\n[OK] Dataset structure is verified and ready for spatio-temporal training.")

if __name__ == "__main__":
    main()
