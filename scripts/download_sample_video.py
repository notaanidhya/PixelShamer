"""
scripts/download_sample_video.py
================================
Utility to prepare test video clips for deepfake video detection.
Supports:
  1. YouTube downloading via yt-dlp
  2. Automatic face-sequence video compilation from existing dataset manifests
  3. Geometric synthetic video fallback (100% offline)
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd

def check_ytdlp():
    try:
        import yt_dlp
        return True
    except ImportError:
        print("[*] Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        return True

def download_clip(url: str, output_name: str, duration_sec: int = 15, category: str = "samples") -> Path | None:
    check_ytdlp()
    import yt_dlp

    base_data = Path(__file__).resolve().parent.parent / "data"
    if category in ("real", "fake"):
        output_dir = base_data / "video_clips" / category
    else:
        output_dir = base_data / "video_samples"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / f"{output_name}.%(ext)s")

    ydl_opts = {
        'format': 'best',
        'outtmpl': out_template,
        'quiet': False,
        'noplaylist': True,
        'download_ranges': yt_dlp.utils.download_range_func(None, [(0, duration_sec)]),
        'force_keyframes_at_cuts': True,
    }

    print(f"[*] Downloading {duration_sec}s clip ({category}) from: {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"[OK] Downloaded to {output_dir}")
        return output_dir
    except Exception as e:
        print(f"[!] YouTube download failed for {url}: {e}")
        return None

def create_clips_from_manifest(
    manifest_csv: Path,
    output_dir: Path,
    label: int,
    num_clips: int = 5,
    frames_per_clip: int = 16
) -> int:
    """Creates realistic video clips by chaining face images from the dataset."""
    if not manifest_csv.exists():
        return 0

    try:
        df = pd.read_csv(manifest_csv)
    except Exception:
        return 0

    subset = df[df["label"] == label]
    if len(subset) < frames_per_clip:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    file_list = list(subset["filepath"].values)
    np.random.seed(42)
    np.random.shuffle(file_list)

    clips_created = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 288, 288

    for c_idx in range(num_clips):
        start_idx = c_idx * frames_per_clip
        if start_idx + frames_per_clip > len(file_list):
            break

        clip_images = file_list[start_idx : start_idx + frames_per_clip]
        clip_name = f"dataset_clip_{c_idx+1:02d}.mp4"
        clip_path = output_dir / clip_name

        writer = cv2.VideoWriter(str(clip_path), fourcc, 10.0, (w, h))
        valid_frames = 0

        for img_path in clip_images:
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_resized = cv2.resize(img, (w, h))
            writer.write(img_resized)
            valid_frames += 1

        writer.release()
        if valid_frames >= frames_per_clip // 2:
            clips_created += 1

    return clips_created

def generate_synthetic_video(output_path: Path, is_fake: bool = False, duration_sec: int = 4, fps: int = 10):
    """Generates a fallback geometric face simulation video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 288, 288
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    center_x, center_y = w // 2, h // 2

    for i in range(total_frames):
        frame = np.full((h, w, 3), (45, 45, 45), dtype=np.uint8)
        offset_x = int(np.sin(i / 6.0) * 12)
        offset_y = int(np.cos(i / 6.0) * 8)
        fx, fy = center_x + offset_x, center_y + offset_y

        # Face oval
        cv2.ellipse(frame, (fx, fy), (65, 85), 0, 0, 360, (185, 195, 220), -1)
        # Eyes
        cv2.circle(frame, (fx - 25, fy - 18), 10, (50, 50, 50), -1)
        cv2.circle(frame, (fx + 25, fy - 18), 10, (50, 50, 50), -1)
        # Mouth
        mouth_y = fy + 32
        if is_fake and i % 4 < 2:
            cv2.ellipse(frame, (fx, mouth_y + 4), (20, 14), 0, 0, 360, (40, 40, 180), -1)
        else:
            cv2.ellipse(frame, (fx, mouth_y), (22, 9), 0, 0, 360, (50, 50, 160), -1)

        out.write(frame)

    out.release()
    print(f"[OK] Generated synthetic video at: {output_path}")

def setup_demo_clips():
    """Builds ready-to-train real and fake video clips from dataset or synthesis."""
    print("=" * 60)
    print("   Setting up Video Clips for Deepfake Detection")
    print("=" * 60)

    base_data = Path(__file__).resolve().parent.parent / "data"
    real_dir = base_data / "video_clips" / "real"
    fake_dir = base_data / "video_clips" / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    manifest_train = base_data / "deepfake" / "manifest_train.csv"
    manifest_test = base_data / "deepfake" / "manifest_test.csv"
    source_manifest = manifest_test if manifest_test.exists() else manifest_train

    real_created = 0
    fake_created = 0

    # 1. Try to compile real and fake video sequences from the dataset
    if source_manifest and source_manifest.exists():
        print(f"[*] Compiling face video clips from dataset: {source_manifest}...")
        real_created = create_clips_from_manifest(source_manifest, real_dir, label=0, num_clips=6)
        fake_created = create_clips_from_manifest(source_manifest, fake_dir, label=1, num_clips=6)
        print(f"[OK] Compiled from dataset: {real_created} real clips, {fake_created} fake clips.")

    # 2. If no dataset clips created, fall back to geometric synthesis
    if real_created == 0 or len(list(real_dir.glob("*.mp4"))) == 0:
        print("[*] Generating synthetic real video clips...")
        for i in range(4):
            generate_synthetic_video(real_dir / f"synthetic_real_{i+1:02d}.mp4", is_fake=False)

    if fake_created == 0 or len(list(fake_dir.glob("*.mp4"))) == 0:
        print("[*] Generating synthetic fake video clips...")
        for i in range(4):
            generate_synthetic_video(fake_dir / f"synthetic_fake_{i+1:02d}.mp4", is_fake=True)

    # 3. Generate Video Manifest
    from ml.deepfake.video_dataset import build_video_manifest
    source_dir = base_data / "video_clips"
    manifest_csv = base_data / "deepfake" / "video_manifest.csv"

    print("\n[*] Generating video manifest from clips...")
    df = build_video_manifest(source_dir, manifest_csv)
    print("=" * 60)
    print(f"[OK] Video Dataset Setup Complete! Found {len(df)} total video clips.")
    print("     You can now run: .\\scripts\\run_phase2.ps1")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup-clips":
        setup_demo_clips()
    elif len(sys.argv) > 1:
        target_url = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) > 2 else "sample_clip"
        cat = sys.argv[3] if len(sys.argv) > 3 else "samples"
        download_clip(target_url, name, category=cat)
    else:
        print("Usage:")
        print("  1. Automatically generate real & fake clips + build manifest:")
        print("     python scripts/download_sample_video.py --setup-clips")
        print("\n  2. Download custom clip into real or fake folder:")
        print("     python scripts/download_sample_video.py <url> <filename> [real|fake]")
