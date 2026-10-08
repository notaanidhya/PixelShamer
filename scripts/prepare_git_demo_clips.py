"""
scripts/prepare_git_demo_clips.py
===================================
Run this script on your TRAINING PC.
It will:
1. Discover genuine video clips from your downloaded 9GB Celeb-DF dataset.
2. Select 6 high-quality Real clips and 6 high-quality Deepfake clips.
3. Transcode/optimize each clip to web-standard H.264 (avc1) at ~1-2 MB each.
4. Place them into 'demo_clips/real/' and 'demo_clips/fake/' so they can be git-pushed directly to GitHub.
5. Offer 1-click git push commands.
"""

import os
import sys
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "demo_clips"
REAL_DEMO = DEMO_DIR / "real"
FAKE_DEMO = DEMO_DIR / "fake"

def is_valid_clip(filepath, min_frames=45, max_frames=600):
    try:
        if not os.path.isfile(filepath):
            return False
        sz = os.path.getsize(filepath)
        if sz < 100 * 1024 or sz > 200 * 1024 * 1024:
            return False
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            return False
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return count >= min_frames and count <= max_frames and w >= 200 and h >= 200
    except Exception:
        return False

def search_candidate_videos():
    search_dirs = [
        Path.home() / "Downloads",
        BASE_DIR / "data",
        BASE_DIR / "data" / "deepfake",
        BASE_DIR / "data" / "celeb_df",
        Path.home() / ".cache" / "kagglehub",
        Path("C:/Users/vitbopal/Downloads"),
    ]
    
    print("[*] Searching for downloaded 9GB deepfake dataset on Training PC...")
    found_real = []
    found_fake = []
    
    for base in search_dirs:
        if not base.exists():
            continue
        print(f"  - Searching in: {base}")
        for root, dirs, files in os.walk(base):
            p_str = root.lower().replace("\\", "/")
            parent_name = Path(root).name.lower()
            
            # Specific dataset folder checks (prevents matching parent folder 'deepfake project')
            is_real_dir = (
                "celeb-real" in p_str
                or "youtube-real" in p_str
                or parent_name in ["real", "original", "actors", "celeb-real", "youtube-real"]
            )
            is_fake_dir = (
                "celeb-synthesis" in p_str
                or parent_name in ["fake", "manipulated", "synthesis", "celeb-synthesis", "faceswap"]
            )
            
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov")):
                    full = Path(root) / f
                    lower_f = f.lower()
                    
                    if is_real_dir or any(k in lower_f for k in ["real", "orig"]):
                        if str(full) not in [str(x) for x in found_real] and is_valid_clip(full):
                            found_real.append(full)
                    elif is_fake_dir or any(k in lower_f for k in ["fake", "df_", "synth"]):
                        if str(full) not in [str(x) for x in found_fake] and is_valid_clip(full):
                            found_fake.append(full)
                            
            if len(found_real) >= 20 and len(found_fake) >= 20:
                break
                
    return found_real, found_fake

def transcode_clip(src_path: Path, dst_path: Path, max_duration_sec: float = 8.0):
    """Transcodes video to H.264 (avc1) web-compatible format, trimmed to ~8s for fast git push."""
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_duration_sec)
    
    # Scale down if 1080p+ to keep file size < 2MB
    target_w, target_h = w, h
    if target_h > 720:
        target_w = int(w * (720 / h))
        target_h = 720
    # Ensure even dimensions for video codecs
    target_w -= target_w % 2
    target_h -= target_h % 2

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(dst_path), fourcc, fps, (target_w, target_h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(dst_path), fourcc, fps, (target_w, target_h))
        
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        if (target_w, target_h) != (w, h):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    return dst_path.exists() and dst_path.stat().st_size > 1024

def main():
    REAL_DEMO.mkdir(parents=True, exist_ok=True)
    FAKE_DEMO.mkdir(parents=True, exist_ok=True)
    
    reals, fakes = search_candidate_videos()
    print(f"[*] Found {len(reals)} real and {len(fakes)} fake candidates.")
    
    if len(reals) == 0 or len(fakes) == 0:
        print("[!] Could not locate Celeb-DF videos in standard folders.")
        print("    Please check the directory where you downloaded the 9GB dataset.")
        return
        
    selected_reals = reals[:6]
    selected_fakes = fakes[:6]
    
    print("\n[*] Transcoding and packaging clips into 'demo_clips/' for Git tracking...")
    total_bytes = 0
    for idx, src in enumerate(selected_reals, start=1):
        dst = REAL_DEMO / f"celeb_real_{idx:02d}.mp4"
        success = transcode_clip(src, dst)
        if success:
            sz = dst.stat().st_size / (1024 * 1024)
            total_bytes += dst.stat().st_size
            print(f"  [+] Real {idx}/6: {src.name} -> {dst.name} ({sz:.2f} MB)")
            
    for idx, src in enumerate(selected_fakes, start=1):
        dst = FAKE_DEMO / f"celeb_fake_{idx:02d}.mp4"
        success = transcode_clip(src, dst)
        if success:
            sz = dst.stat().st_size / (1024 * 1024)
            total_bytes += dst.stat().st_size
            print(f"  [+] Fake {idx}/6: {src.name} -> {dst.name} ({sz:.2f} MB)")
            
    total_mb = total_bytes / (1024 * 1024)
    print("\n" + "=" * 65)
    print(f"[SUCCESS] Prepared {len(selected_reals) + len(selected_fakes)} genuine demo clips ({total_mb:.1f} MB total).")
    print("=" * 65)
    print("\nNow run these commands on your TRAINING PC to push them via Git:")
    print("  git add demo_clips/")
    print("  git commit -m \"add genuine Celeb-DF demo clips\"")
    print("  git push origin main")
    print("\nThen on your DEV PC, simply run:")
    print("  git pull origin main")

if __name__ == "__main__":
    main()
