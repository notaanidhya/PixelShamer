"""
scripts/package_training_clips.py
==================================
Run this script on your TRAINING PC to automatically find genuine video clips
from your downloaded 9GB deepfake dataset, select 10 high-quality Real videos
and 10 high-quality Deepfake videos, and package them into 'genuine_demo_clips.zip'.
"""

import os
import sys
import glob
import zipfile
import cv2

def is_valid_video_clip(filepath, min_frames=45, max_frames=600):
    """Checks if a video file is a valid, readable video with adequate length."""
    try:
        if not os.path.isfile(filepath):
            return False
        sz = os.path.getsize(filepath)
        if sz < 100 * 1024 or sz > 150 * 1024 * 1024: # between 100KB and 150MB
            return False
        cap = cv2.VideoCapture(filepath)
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
        os.path.join(os.path.expanduser("~"), "Downloads"),
        "data",
        os.path.join("data", "deepfake"),
        os.path.join(os.path.expanduser("~"), ".cache", "kagglehub"),
        "C:\\Users\\vitbopal\\Downloads",
    ]
    
    print("[*] Searching for downloaded 9GB deepfake video dataset on Training PC...")
    found_real = []
    found_fake = []
    
    for base in search_dirs:
        if not os.path.exists(base):
            continue
        print(f"  - Searching in: {base}")
        for root, dirs, files in os.walk(base):
            # Check directory names
            lower_root = root.lower()
            is_real_dir = any(k in lower_root for k in ["real", "original", "actors", "youtube"])
            is_fake_dir = any(k in lower_root for k in ["fake", "manipulated", "deepfake", "faceswap", "neuraltextures", "face2face", "face_swap", "df"])
            
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov")):
                    full = os.path.join(root, f)
                    lower_f = f.lower()
                    
                    if is_fake_dir or any(k in lower_f for k in ["fake", "df_", "synth"]):
                        if full not in found_fake and is_valid_video_clip(full):
                            found_fake.append(full)
                    elif is_real_dir or any(k in lower_f for k in ["real", "orig"]):
                        if full not in found_real and is_valid_video_clip(full):
                            found_real.append(full)
                            
            if len(found_real) >= 30 and len(found_fake) >= 30:
                break
                
    return found_real, found_fake

def package_clips(real_clips, fake_clips, output_zip="genuine_demo_clips.zip"):
    selected_real = real_clips[:10]
    selected_fake = fake_clips[:10]
    
    if len(selected_real) == 0 and len(selected_fake) == 0:
        print("[!] No video clips found in standard download locations.")
        print("    Please check the directory path where your 9GB dataset is saved.")
        return False
        
    print(f"[*] Packaging {len(selected_real)} Real clips and {len(selected_fake)} Deepfake clips...")
    
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, path in enumerate(selected_real, start=1):
            arcname = f"real/genuine_real_{idx:02d}.mp4"
            zf.write(path, arcname)
            print(f"  + Added Real [{idx}]: {os.path.basename(path)}")
            
        for idx, path in enumerate(selected_fake, start=1):
            arcname = f"fake/genuine_fake_{idx:02d}.mp4"
            zf.write(path, arcname)
            print(f"  + Added Fake [{idx}]: {os.path.basename(path)}")
            
    zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print("=" * 65)
    print(f"[SUCCESS] Packaged demo dataset into: {output_zip} ({zip_size_mb:.1f} MB)")
    print("=" * 65)
    print("Next step:")
    print(f"1. Download '{output_zip}' from this Training PC to your Dev PC.")
    print("2. On Dev PC, run: venv\\Scripts\\python scripts/install_demo_clips.py")
    return True

if __name__ == "__main__":
    reals, fakes = search_candidate_videos()
    print(f"[*] Discovered {len(reals)} real video candidates, {len(fakes)} fake video candidates.")
    package_clips(reals, fakes)
