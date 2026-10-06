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

def download_clip(url: str, output_name: str, duration_sec: int = 15):
    check_ytdlp()
    import yt_dlp

    output_dir = Path(__file__).resolve().parent.parent / "data" / "video_samples"
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

    print(f"[*] Downloading {duration_sec}s clip from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(f"[OK] Downloaded to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) > 2 else "sample_clip"
        download_clip(target_url, name)
    else:
        print("Usage: python scripts/download_sample_video.py <youtube_url> [output_name]")
        print("\nPre-configured examples you can run:")
        print("  python scripts/download_sample_video.py https://www.youtube.com/watch?v=oxXpB9pSETo fake_cruise")
        print("  python scripts/download_sample_video.py https://www.youtube.com/watch?v=cQ54GDm1eL0 fake_obama")
