"""
scripts/install_demo_clips.py
==============================
Run this script on DEV PC after copying 'genuine_demo_clips.zip' from the Training PC.
Replaces the old rapid image-swap clips in data/video_clips/ with genuine video footage.
"""

import os
import shutil
import zipfile
import subprocess
import glob
import imageio_ffmpeg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(BASE_DIR, "genuine_demo_clips.zip")
CLIPS_DIR = os.path.join(BASE_DIR, "data", "video_clips")

def main():
    if not os.path.exists(ZIP_PATH):
        print(f"[!] '{ZIP_PATH}' not found.")
        print("    Please copy 'genuine_demo_clips.zip' from your Training PC into the project root.")
        return

    print("[*] Found 'genuine_demo_clips.zip'. Preparing to install genuine demo clips...")
    
    # Clean old synthetic video files
    for sub in ["real", "fake"]:
        target_dir = os.path.join(CLIPS_DIR, sub)
        if os.path.exists(target_dir):
            for f in glob.glob(os.path.join(target_dir, "*.*")):
                try:
                    os.remove(f)
                except Exception:
                    pass
        os.makedirs(target_dir, exist_ok=True)
        
    # Extract zip
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(CLIPS_DIR)
    print(f"[*] Extracted clips into: {CLIPS_DIR}")
    
    # Transcode to H.264
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    all_extracted = glob.glob(os.path.join(CLIPS_DIR, "*", "*.mp4"))
    print(f"[*] Transcoding {len(all_extracted)} clips to web-standard H.264...")
    
    for clip in all_extracted:
        tmp = clip + ".tmp.mp4"
        cmd = [
            ffmpeg_exe, "-y", "-v", "error",
            "-i", clip,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            tmp
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, clip)
            
    print("=" * 65)
    print(f"[SUCCESS] Genuine video clips installed into data/video_clips/!")
    print(f"  - Real clips: {len(glob.glob(os.path.join(CLIPS_DIR, 'real', '*.mp4')))}")
    print(f"  - Fake clips: {len(glob.glob(os.path.join(CLIPS_DIR, 'fake', '*.mp4')))}")
    print("=" * 65)

if __name__ == "__main__":
    main()
