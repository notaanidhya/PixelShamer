"""
scripts/download_deepfake_dataset.py
====================================
Automated Downloader & Preprocessor for Real and Deepfake Video Datasets.

Sources supported:
  1. 'hf' (Default / Recommended):
     Downloads the curated SDFVD benchmark (106 high-quality clips: 53 Real, 53 Deepfake).
     Fast (~130 MB total), completely public (no API keys, tokens, or Kaggle login required).
  2. 'celeb-df':
     Downloads Celeb-DF v2 via Kaggle API (~9.9 GB, 5,600+ deepfake sequences).
  3. 'ff':
     Downloads FaceForensics++ (C23 compressed) via Kaggle API (~17 GB).

Usage:
  # Download the 106 real & fake video clips (fastest & easiest):
  python scripts/download_deepfake_dataset.py

  # Download first 10 real and 10 fake clips only:
  python scripts/download_deepfake_dataset.py --limit 10

  # Download Celeb-DF v2 (requires Kaggle CLI / credentials):
  python scripts/download_deepfake_dataset.py --source celeb-df
"""

import os
import sys
import json
import argparse
import subprocess
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.deepfake.video_dataset import build_video_manifest

HF_REPO_API = "https://huggingface.co/api/datasets/Hemgg/SDFVD-video-dataset"
HF_BASE_URL = "https://huggingface.co/datasets/Hemgg/SDFVD-video-dataset/resolve/main"

def download_file(url: str, dest_path: Path) -> tuple[str, bool, int]:
    """Downloads a single file from URL to dest_path."""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and dest_path.stat().st_size > 10000:
            return (dest_path.name, True, dest_path.stat().st_size)

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DeepfakeForensics/1.0 (Mozilla/5.0)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out_file:
            bytes_written = 0
            while chunk := resp.read(65536):
                out_file.write(chunk)
                bytes_written += len(chunk)
        return (dest_path.name, True, bytes_written)
    except Exception as e:
        print(f"[!] Error downloading {url}: {e}")
        return (dest_path.name, False, 0)

def fetch_hf_dataset(target_dir: Path, limit: int | None = None, max_workers: int = 6):
    """
    Downloads the SDFVD (Small Deepfake Video Dataset) from Hugging Face.
    53 real clips and 53 deepfake clips.
    """
    print("=" * 65)
    print("   Fetching DeepFake Video Dataset: SDFVD (Hugging Face)")
    print("=" * 65)
    print(f"[*] Querying dataset catalog from: {HF_REPO_API}")

    req = urllib.request.Request(HF_REPO_API, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            metadata = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[!] Failed to fetch repository index from Hugging Face: {e}")
        return False

    siblings = metadata.get("siblings", [])
    real_files = [s["rfilename"] for s in siblings if s["rfilename"].startswith("Real/") and s["rfilename"].endswith(".mp4")]
    fake_files = [s["rfilename"] for s in siblings if s["rfilename"].startswith("Fake/") and s["rfilename"].endswith(".mp4")]

    # Sort numerically (v1, v2, v3...)
    def sort_key(fn: str):
        stem = Path(fn).stem.replace("vs", "").replace("v", "")
        return int(stem) if stem.isdigit() else 9999

    real_files.sort(key=sort_key)
    fake_files.sort(key=sort_key)

    if limit is not None and limit > 0:
        real_files = real_files[:limit]
        fake_files = fake_files[:limit]

    print(f"[*] Found {len(real_files)} Real clips and {len(fake_files)} Fake clips to download.")
    print(f"[*] Destination: {target_dir.resolve()}")

    tasks = []
    real_out = target_dir / "real"
    fake_out = target_dir / "fake"

    for r_fn in real_files:
        url = f"{HF_BASE_URL}/{r_fn}"
        dest = real_out / Path(r_fn).name
        tasks.append((url, dest, "real"))

    for f_fn in fake_files:
        url = f"{HF_BASE_URL}/{f_fn}"
        dest = fake_out / Path(f_fn).name
        tasks.append((url, dest, "fake"))

    total_tasks = len(tasks)
    completed = 0
    total_bytes = 0

    print(f"[*] Initiating multi-threaded download ({max_workers} worker threads)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(download_file, url, dest): (dest.name, cat) for url, dest, cat in tasks}
        for future in as_completed(future_map):
            fname, cat = future_map[future]
            completed += 1
            name, success, size = future.result()
            total_bytes += size
            mb_downloaded = total_bytes / (1024 * 1024)
            pct = (completed / total_tasks) * 100
            status_str = f"[{completed}/{total_tasks} - {pct:5.1f}%] {cat.upper()}: {fname} ({size / (1024*1024):.1f} MB | Total: {mb_downloaded:.1f} MB)"
            print(f"\r{status_str}", end="", flush=True)

    print("\n[OK] All clips downloaded successfully!")
    return True

def fetch_kaggle_dataset(dataset_ref: str, target_dir: Path):
    """Downloads large benchmark datasets from Kaggle via the kaggle CLI."""
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_videos = list(target_dir.rglob("*.mp4"))
    if len(existing_videos) > 50:
        print(f"[*] Found {len(existing_videos)} video files already extracted in: {target_dir}")
        print("    Skipping download to avoid redundant network transfer.")
        return True

    print("=" * 65)
    print(f"   Downloading Kaggle Dataset: {dataset_ref}")
    print("=" * 65)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(target_dir), "--unzip"]
    print(f"[*] Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"[OK] Successfully downloaded and extracted {dataset_ref} into {target_dir}")
        return True
    except FileNotFoundError:
        print("[!] Error: 'kaggle' CLI is not installed or not in PATH.")
        print("    Install it via: pip install kaggle")
        print("    And place your kaggle.json in ~/.kaggle/kaggle.json")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[!] Kaggle download failed with exit code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download real and fake deepfake video datasets.")
    parser.add_argument(
        "--source",
        choices=["hf", "celeb-df", "ff"],
        default="hf",
        help="Dataset source: 'hf' (SDFVD 106 clips, default), 'celeb-df' (Celeb-DF v2 via Kaggle), 'ff' (FaceForensics++ via Kaggle)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of real and fake clips to download (e.g. --limit 500 for fast high-accuracy training)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="Number of concurrent download worker threads (default: 6)"
    )
    args = parser.parse_args()

    base_data = ROOT_DIR / "data"
    clips_dir = base_data / "video_clips"
    manifest_csv = base_data / "deepfake" / "video_manifest.csv"

    if args.source == "hf":
        target_dir = clips_dir
        success = fetch_hf_dataset(target_dir, limit=args.limit, max_workers=args.threads)
    elif args.source == "celeb-df":
        target_dir = base_data / "celeb_df"
        success = fetch_kaggle_dataset("reubensuju/celeb-df-v2", target_dir)
    elif args.source == "ff":
        target_dir = base_data / "faceforensics"
        success = fetch_kaggle_dataset("xdxd003/ff-c23", target_dir)

    if success:
        print("\n" + "=" * 65)
        print("   Generating Spatio-Temporal Video Manifest")
        print("=" * 65)
        df = build_video_manifest(target_dir, manifest_csv, limit=args.limit, balance=True)
        real_count = len(df[df["label"] == 0])
        fake_count = len(df[df["label"] == 1])
        print(f"[OK] Manifest created with {len(df)} total clips ({real_count} Real, {fake_count} Fake).")
        print(f"     Saved to: {manifest_csv}")
        print("\nNext step: Run Phase 2 model training on the GPU:")
        print("  .\\scripts\\run_phase2.ps1")
        print("=" * 65)

if __name__ == "__main__":
    main()
