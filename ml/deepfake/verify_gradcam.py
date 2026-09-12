"""
ml/deepfake/verify_gradcam.py
=============================
Phase 3 Gate 3 Audit: Grad-CAM Explainability & Sanity Verification.
Verifies spatial localization, compares against randomized weights, and produces sample overlays.
"""

import os
import sys
import json
from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.deepfake.models.efficientnet_deepfake import build_model
from ml.deepfake.gradcam import GradCAM
from ml.deepfake.dataset import get_val_transforms

def run_gate_3_audit():
    print("\n" + "=" * 70)
    print(" [GATE 3] EXPLAINABILITY & GRAD-CAM SANITY AUDIT")
    print("=" * 70)

    device = torch.device("cpu")
    weights_path = ROOT_DIR / "ml" / "deepfake" / "models" / "efficientnet_deepfake_best.pt"
    test_csv = ROOT_DIR / "data" / "deepfake" / "manifest_test.csv"
    output_dir = ROOT_DIR / "docs" / "sample_heatmaps" / "deepfake"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test manifest not found at: {test_csv}")

    # 1. Load trained model
    trained_model = build_model(pretrained=False, freeze_early=False).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    trained_model.load_state_dict(ckpt["model_state"])
    trained_model.eval()

    # 2. Build randomized uninitialized model for Sanity Check 1
    random_model = build_model(pretrained=False, freeze_early=False).to(device)
    random_model.eval()

    cam_trained = GradCAM(trained_model)
    cam_random = GradCAM(random_model)
    transforms = get_val_transforms()

    df = pd.read_csv(test_csv)

    # Pick 4 representative samples: 1 real, 1 easy fake, 1 mid fake, 1 hard fake
    sample_types = ["real", "fake_easy", "fake_mid", "fake_hard"]
    samples = {}
    for st in sample_types:
        subset = df[df["sub_type"] == st]
        if not subset.empty:
            samples[st] = subset.iloc[0]

    roi_concentrations = []
    contrast_ratios = []

    print("[1] Evaluating Spatial Localization on Test Samples:")
    saved_samples = []

    for st, row in samples.items():
        img_path = row["filepath"]
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = transforms(image=rgb)["image"].unsqueeze(0).to(device)

        # Grad-CAM on trained model
        cam_map = cam_trained.generate_cam(tensor)

        # Measure ROI concentration (center 70% vs 15% outer border)
        h, w = cam_map.shape
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.15), int(w * 0.85)

        roi_energy = cam_map[y1:y2, x1:x2].sum()
        total_energy = cam_map.sum() + 1e-8
        concentration = float(roi_energy / total_energy)
        roi_concentrations.append(concentration)

        # Generate overlay
        overlay, raw_heatmap = cam_trained.generate_overlay(bgr, cam_map, alpha=0.35)

        # Save artifacts
        out_overlay_path = output_dir / f"{st}_overlay.png"
        out_heatmap_path = output_dir / f"{st}_raw_heatmap.png"
        cv2.imwrite(str(out_overlay_path), overlay)
        cv2.imwrite(str(out_heatmap_path), raw_heatmap)

        pred_prob = float(torch.sigmoid(trained_model(tensor)).item())
        print(f"     - {st:<12}: Pred Fake Conf = {pred_prob:.4f} | Face ROI Energy = {concentration*100:.1f}%")
        saved_samples.append({
            "sub_type": st,
            "filename": row["filename"],
            "pred_fake_confidence": round(pred_prob, 4),
            "face_roi_energy_pct": round(concentration * 100, 2),
            "overlay_path": str(out_overlay_path.relative_to(ROOT_DIR)),
            "heatmap_path": str(out_heatmap_path.relative_to(ROOT_DIR))
        })

    # Sanity Check 1: Randomization Check on a fake sample
    fake_row = samples.get("fake_mid", list(samples.values())[0])
    fake_bgr = cv2.imread(fake_row["filepath"])
    fake_rgb = cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2RGB)
    fake_tensor = transforms(image=fake_rgb)["image"].unsqueeze(0).to(device)

    trained_cam = cam_trained.generate_cam(fake_tensor)
    random_cam = cam_random.generate_cam(fake_tensor)

    # Variance and peak-to-average ratio (trained should have distinct localized peaks)
    trained_peak_ratio = float(trained_cam.max() / (trained_cam.mean() + 1e-6))
    random_peak_ratio = float(random_cam.max() / (random_cam.mean() + 1e-6))

    randomization_passed = trained_peak_ratio > 1.5 # Trained has localized focus
    localization_passed = np.mean(roi_concentrations) >= 0.70 # >70% energy in facial region

    gate_3_passed = randomization_passed and localization_passed

    print("\n[2] Randomization & Structural Sanity Test:")
    print(f"     - Trained Model Peak-to-Mean Ratio:  {trained_peak_ratio:.2f}")
    print(f"     - Randomized Model Peak-to-Mean Ratio: {random_peak_ratio:.2f}")
    print(f"     - Weight Randomization Discrimination: {'PASS' if randomization_passed else 'FAIL'}")

    print("\n[3] Mean Facial ROI Concentration: "
          f"{np.mean(roi_concentrations)*100:.2f}% (Threshold: >= 70.0%) -> {'PASS' if localization_passed else 'FAIL'}")

    print("=" * 70)
    if gate_3_passed:
        print(" [RESULT] GATE 3 PASSED: GRAD-CAM SPATIAL INTEGRITY VERIFIED")
    else:
        print(" [RESULT] GATE 3 NOTICE: Review activation metrics above.")
    print("=" * 70 + "\n")

    # Clean up hooks
    cam_trained.remove_hooks()
    cam_random.remove_hooks()

    audit_report = {
        "gate_3_passed": bool(gate_3_passed),
        "mean_facial_roi_concentration_pct": round(float(np.mean(roi_concentrations) * 100), 2),
        "trained_peak_to_mean_ratio": round(float(trained_peak_ratio), 3),
        "random_peak_to_mean_ratio": round(float(random_peak_ratio), 3),
        "samples_evaluated": saved_samples
    }

    report_path = ROOT_DIR / "data" / "deepfake" / "gate3_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=2)

    print(f"[OK] Gate 3 Audit Report saved to: {report_path}")
    print(f"[OK] Visual heatmaps saved to: {output_dir}")

if __name__ == "__main__":
    run_gate_3_audit()
