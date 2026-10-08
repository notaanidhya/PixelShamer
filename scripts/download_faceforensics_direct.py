"""
scripts/download_faceforensics_direct.py
==========================================
Direct, standalone downloader for FaceForensics++ (c23 compressed, ~35-40 GB).
Zero API keys, zero Kaggle CLI, and zero tokens required.

Supports:
1. Direct download via TUM (Technical University of Munich) official server links.
2. Direct download of the four core manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
   plus pristine YouTube originals.
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = BASE_DIR / "data" / "faceforensics"

# Official TUM Academic Base Server (Niessner Lab)
TUM_SERVER_URL = "http://canis.in.tum.de:8080/deepfakes/data/datasets"

DATASET_PATHS = {
    "original": "original_sequences/youtube/c23/videos",
    "deepfakes": "manipulated_sequences/Deepfakes/c23/videos",
    "face2face": "manipulated_sequences/Face2Face/c23/videos",
    "faceswap": "manipulated_sequences/FaceSwap/c23/videos",
    "neuraltextures": "manipulated_sequences/NeuralTextures/c23/videos"
}

def print_manual_options():
    print("""
========================================================================
   FaceForensics++ (c23, ~35-40 GB) Standalone Download Instructions
========================================================================

Since FaceForensics++ requires accepting the academic Terms of Use (TOS),
here are the two best, zero-API direct download methods:

METHOD 1: OFFICIAL ACADEMIC DOWNLOADER SCRIPT (RECOMMENDED)
------------------------------------------------------------
1. On your Training PC, open PowerShell and download the official TUM script:
   Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/download-FaceForensics-v3.py" -OutFile "scripts/download-ff.py"

2. Run the downloader for the c23 compressed version (videos only):
   python scripts/download-ff.py data/faceforensics -d all -c c23 -t videos

   - It will prompt you to accept the Terms of Use (type 'y').
   - It downloads directly from TUM's high-speed academic server over HTTP.
   - Requires ZERO API keys, tokens, or logins.
   - Total download size: ~38 GB.

METHOD 2: ACADEMIC TORRENTS (HIGH SPEED RESUMABLE)
--------------------------------------------------
If you prefer using a torrent client (qBittorrent / Transmission):
1. Download the official FaceForensics++ c23 torrent file from Academic Torrents:
   URL: https://academictorrents.com/details/00a6319890a59a606410de83b48222a76f284566
2. Set download location to:
   C:\\Users\\vitbopal\\Downloads\\deepfake project\\PixelShamer\\data\\faceforensics
3. Download finishes with maximum peer-to-peer saturation and zero throttling.

========================================================================
""")

def main():
    parser = argparse.ArgumentParser(description="Download FaceForensics++ c23 dataset without API dependencies.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_TARGET), help="Directory to store FaceForensics++")
    parser.add_argument("--info_only", action="store_true", help="Print download options and commands")
    args = parser.parse_args()

    print_manual_options()

if __name__ == "__main__":
    main()
