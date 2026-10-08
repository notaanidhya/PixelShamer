"""
scripts/clean_and_sync_demo_clips.py
======================================
Cleans out the old synthetic image-swap clips from data/video_clips/
and synchronizes with the genuine video clips pulled from git in demo_clips/.
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "demo_clips"
DATA_CLIPS = BASE_DIR / "data" / "video_clips"

def clean_and_sync():
    print("[*] Cleaning up old synthetic image-swap test clips...")
    for sub in ["real", "fake"]:
        target_dir = DATA_CLIPS / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove old synthetic clips (v*.mp4, dataset_clip_*.mp4)
        for f in list(target_dir.glob("v*.mp4")) + list(target_dir.glob("dataset_clip_*.mp4")):
            try:
                f.unlink()
                print(f"  [-] Removed synthetic clip: {f.name}")
            except Exception as e:
                print(f"  [!] Failed to remove {f.name}: {e}")
                
        # Copy from demo_clips if available
        src_dir = DEMO_DIR / sub
        if src_dir.exists():
            clips = [f for f in src_dir.glob("*.mp4") if not f.name.startswith(".")]
            for clip in clips:
                dest = target_dir / clip.name
                shutil.copy2(clip, dest)
                print(f"  [+] Synced genuine clip to data/video_clips/{sub}/: {clip.name}")

    print("\n[SUCCESS] Synchronization complete!")

if __name__ == "__main__":
    clean_and_sync()
