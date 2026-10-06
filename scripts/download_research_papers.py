"""
scripts/download_research_papers.py
===================================
Automated downloader for curated research papers relevant to the
PixelShamer / DeepFake Detection System project.
Downloads PDFs into the 'research_papers/' directory and validates their format.
"""

import os
import sys
import time
import urllib.request
import urllib.error

TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research_papers")

PAPERS = [
    {
        "filename": "01_FaceForensics++_ICCV2019.pdf",
        "url": "https://arxiv.org/pdf/1901.08971.pdf",
        "title": "FaceForensics++: Learning to Detect Manipulated Facial Images (ICCV 2019)"
    },
    {
        "filename": "02_Face_XRay_CVPR2020.pdf",
        "url": "https://arxiv.org/pdf/1912.13458.pdf",
        "title": "Face X-ray for More General Face Forgery Detection (CVPR 2020 Oral)"
    },
    {
        "filename": "03_CNN_Generated_Images_CVPR2020.pdf",
        "url": "https://arxiv.org/pdf/1912.11035.pdf",
        "title": "CNN-generated images are surprisingly easy to spot... for now (CVPR 2020)"
    },
    {
        "filename": "04_F3Net_Frequency_ECCV2020.pdf",
        "url": "https://arxiv.org/pdf/2007.09355.pdf",
        "title": "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues (ECCV 2020)"
    },
    {
        "filename": "05_CelebDF_CVPR2020.pdf",
        "url": "https://arxiv.org/pdf/1909.12962.pdf",
        "title": "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics (CVPR 2020)"
    },
    {
        "filename": "06_Face_Warping_CVPRW2019.pdf",
        "url": "https://arxiv.org/pdf/1811.00656.pdf",
        "title": "Exposing DeepFake Videos by Detecting Face Warping Artifacts (CVPRW 2019)"
    },
    {
        "filename": "07_MultiAttentional_CVPR2021.pdf",
        "url": "https://arxiv.org/pdf/2103.02406.pdf",
        "title": "Multi-attentional Deepfake Detection (CVPR 2021)"
    },
    {
        "filename": "08_Universal_Detectors_CVPR2023.pdf",
        "url": "https://arxiv.org/pdf/2302.10174.pdf",
        "title": "Towards Universal Fake Image Detectors that Generalize Across Generative Models (CVPR 2023)"
    },
    {
        "filename": "09_DIRE_Diffusion_ICCV2023.pdf",
        "url": "https://arxiv.org/pdf/2303.09295.pdf",
        "title": "DIRE for Diffusion-Generated Image Detection (ICCV 2023)"
    },
    {
        "filename": "10_FakeCatcher_PPG_TPAMI2020.pdf",
        "url": "https://arxiv.org/pdf/1901.02212.pdf",
        "title": "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals (IEEE TPAMI 2020)"
    },
    {
        "filename": "11_GradCAM_ICCV2017.pdf",
        "url": "https://arxiv.org/pdf/1610.02391.pdf",
        "title": "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization (ICCV 2017)"
    },
    {
        "filename": "12_DeepfakeBench_NeurIPS2023.pdf",
        "url": "https://arxiv.org/pdf/2307.01426.pdf",
        "title": "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection (NeurIPS 2023)"
    },
    {
        "filename": "13_BRISQUE_Mittal_TIP2012.pdf",
        "url": "http://live.ece.utexas.edu/publications/2012/mittal_tip_2012.pdf",
        "title": "No-Reference Image Quality Assessment in the Spatial Domain - BRISQUE (IEEE TIP 2012)"
    },
    {
        "filename": "14_MVTec_Anomaly_CVPR2019.pdf",
        "url": "https://arxiv.org/pdf/1905.00940.pdf",
        "title": "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection (CVPR 2019)"
    },
    {
        "filename": "15_Deepfake_Survey_Reliability_2024.pdf",
        "url": "https://arxiv.org/pdf/2409.15180.pdf",
        "title": "Deepfake Detection: A Comprehensive Survey from the Reliability Perspective (2024)"
    }
]

def download_paper(paper_info: dict, out_dir: str) -> bool:
    filename = paper_info["filename"]
    url = paper_info["url"]
    title = paper_info["title"]
    dest_path = os.path.join(out_dir, filename)

    print(f"\n[+] Fetching: {title}")
    print(f"    Source: {url}")
    print(f"    Target: {filename}")

    # Check if already downloaded and valid
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 50_000:
        with open(dest_path, "rb") as f:
            header = f.read(5)
            if header.startswith(b"%PDF-"):
                size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                print(f"    [SKIP] Already exists and valid ({size_mb:.2f} MB)")
                return True

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
    }

    req = urllib.request.Request(url, headers=headers)

    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
                if not data.startswith(b"%PDF-"):
                    print(f"    [!] Warning: Downloaded content does not begin with %PDF- header (attempt {attempt})")
                    if attempt < 3:
                        time.sleep(2)
                        continue
                    return False
                
                with open(dest_path, "wb") as out_f:
                    out_f.write(data)

                size_mb = len(data) / (1024 * 1024)
                print(f"    [SUCCESS] Downloaded {size_mb:.2f} MB")
                return True
        except Exception as err:
            print(f"    [ERROR] Attempt {attempt} failed: {err}")
            if attempt < 3:
                time.sleep(3)

    return False

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"==================================================")
    print(f"Downloading {len(PAPERS)} Research Papers into:")
    print(f"{TARGET_DIR}")
    print(f"==================================================")

    successful = 0
    failed = []

    for idx, paper in enumerate(PAPERS, 1):
        print(f"Progress: [{idx}/{len(PAPERS)}]")
        ok = download_paper(paper, TARGET_DIR)
        if ok:
            successful += 1
        else:
            failed.append(paper["title"])
        # Friendly rate limiting delay
        time.sleep(1.2)

    print("\n==================================================")
    print(f"Download Summary: {successful}/{len(PAPERS)} Successful")
    if failed:
        print("Failed Papers:")
        for f in failed:
            print(f" - {f}")
    print("==================================================")

if __name__ == "__main__":
    main()
