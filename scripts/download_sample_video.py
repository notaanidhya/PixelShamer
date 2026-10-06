"""
scripts/download_sample_video.py
================================
Utility to download short test videos from YouTube or direct URLs.
Extracts and trims clips for deepfake video testing.
Requires: pip install yt-dlp
"""

import sys
import subprocess
from pathlib import Path

DEFAULT_URLS = {
    "fake_tom_cruise": "https://www.youtube.com/watch?v=oxXpB9pSETo",
    "fake_obama": "https://www.youtube.com/watch?v=cQ54GDm1eL0",
    "real_interview": "https://www.youtube.com/watch?v=13Qf9g_c74s"
}

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
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
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
        print(f"[!] Download failed for {url}: {e}")
        return None

def setup_demo_clips():
    """Downloads 1 real and 1 fake clip and generates the manifest automatically."""
    print("=" * 60)
    print("   Setting up Demo Video Clips for Deepfake Detection")
    print("=" * 60)

    # 1. Download Fake (Deepfake Tom Cruise)
    download_clip("https://www.youtube.com/watch?v=oxXpB9pSETo", "fake_tom_cruise", duration_sec=15, category="fake")

    # 2. Download Real (Morgan Freeman interview / speech)
    download_clip("https://www.youtube.com/watch?v=13Qf9g_c74s", "real_speech", duration_sec=15, category="real")

    # 3. Generate Manifest
    from ml.deepfake.video_dataset import build_video_manifest
    base_data = Path(__file__).resolve().parent.parent / "data"
    source_dir = base_data / "video_clips"
    manifest_csv = base_data / "deepfake" / "video_manifest.csv"

    print("\n[*] Generating video manifest from downloaded clips...")
    df = build_video_manifest(source_dir, manifest_csv)
    print("=" * 60)
    print(f"[OK] Setup complete! Found {len(df)} clips in manifest.")
    print("     You can now immediately run: .\\scripts\\run_phase2.ps1")
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
        print("  1. Automatically download real & fake demo clips + build manifest:")
        print("     python scripts/download_sample_video.py --setup-clips")
        print("\n  2. Download custom clip into real or fake folder:")
        print("     python scripts/download_sample_video.py <url> <filename> [real|fake]")
