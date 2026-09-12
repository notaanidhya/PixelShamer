"""
backend/app/services/deepfake_inference.py
==========================================
Inference service for Deepfake Face Forgery Detection.
Handles image ingestion, face localization, EfficientNet-B2 scoring,
Grad-CAM explainability generation, and deterministic spectral telemetry extraction.
"""

import os
import sys
import uuid
import logging
import cv2
import numpy as np
import torch

logger = logging.getLogger("deepfake_api.inference")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.deepfake.models.efficientnet_deepfake import build_model
from ml.deepfake.gradcam import GradCAM
from ml.deepfake.dataset import get_val_transforms
from ml.feature_extractor import extract_features

class DeepfakeInferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.gradcam = None
        self.face_cascade = None
        self.profile_cascade = None
        self.transforms = None
        self.is_ready = False
        self.calibrated_threshold = 0.580

    def load_models(self, model_path: str = None):
        """Loads EfficientNet-B2 weights, initializes Grad-CAM, and prepares face detectors."""
        if model_path is None:
            model_path = os.path.join(BASE_DIR, "ml", "deepfake", "models", "efficientnet_deepfake_best.pt")

        logger.info(f"Loading Deepfake Model from: {model_path} on {self.device}")
        
        self.model = build_model(pretrained=False, freeze_early=False).to(self.device)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state"])
            self.calibrated_threshold = checkpoint.get("optimal_threshold", 0.580)
            logger.info(f"Loaded checkpoint with calibrated threshold: {self.calibrated_threshold:.3f}")
        else:
            logger.warning(f"Checkpoint not found at {model_path}. Running with initialized weights.")

        self.model.eval()
        self.gradcam = GradCAM(self.model)
        self.transforms = get_val_transforms()

        # Load bundled OpenCV Haar Cascades for face detection
        bundled_dir = os.path.join(BASE_DIR, "ml", "deepfake", "models", "haarcascades")
        sys_dir = getattr(cv2.data, "haarcascades", "") if hasattr(cv2, "data") else ""
        
        frontal_path = (
            os.path.join(bundled_dir, "haarcascade_frontalface_default.xml")
            if os.path.exists(os.path.join(bundled_dir, "haarcascade_frontalface_default.xml"))
            else os.path.join(sys_dir, "haarcascade_frontalface_default.xml")
        )
        profile_path = (
            os.path.join(bundled_dir, "haarcascade_profileface.xml")
            if os.path.exists(os.path.join(bundled_dir, "haarcascade_profileface.xml"))
            else os.path.join(sys_dir, "haarcascade_profileface.xml")
        )

        if os.path.exists(frontal_path):
            self.face_cascade = cv2.CascadeClassifier(frontal_path)
            logger.info(f"Loaded frontal face cascade from: {frontal_path}")
        else:
            logger.warning(f"Frontal face cascade XML not found at {frontal_path}")

        if os.path.exists(profile_path):
            self.profile_cascade = cv2.CascadeClassifier(profile_path)
            logger.info(f"Loaded profile face cascade from: {profile_path}")
        else:
            self.profile_cascade = None

        self.is_ready = True
        logger.info("DeepfakeInferenceService initialized successfully.")

    def detect_face(self, image_bgr: np.ndarray) -> tuple[np.ndarray | None, dict | None]:
        """
        Localizes face in the image and returns an expanded crop (1.2x margin).
        Uses frontal cascade first, falls back to profile cascade (both orientations).
        Only falls back to full image if image is already a square-ish close-up portrait.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        detected_faces = []

        # 1. Frontal face detection
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=(60, 60)
            )
            if len(faces) > 0:
                detected_faces.extend(faces)

        # 2. Profile face detection (if frontal missed face, e.g. camera angled/tilted)
        if len(detected_faces) == 0 and self.profile_cascade is not None and not self.profile_cascade.empty():
            prof_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(60, 60)
            )
            if len(prof_faces) > 0:
                detected_faces.extend(prof_faces)
            else:
                # Try flipped for opposite profile orientation
                flipped_gray = cv2.flip(gray, 1)
                prof_flipped = self.profile_cascade.detectMultiScale(
                    flipped_gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(60, 60)
                )
                if len(prof_flipped) > 0:
                    for (fx, fy, fw, fh) in prof_flipped:
                        detected_faces.append((w - fx - fw, fy, fw, fh))

        # 3. Handle when no face is localized by cascades
        if len(detected_faces) == 0:
            aspect = w / float(h)
            # Only treat as already-cropped face portrait if dimensions and aspect ratio match standard portrait crops
            if 0.80 <= aspect <= 1.25 and max(h, w) <= 800 and min(h, w) >= 120:
                return image_bgr, {"x": 0, "y": 0, "w": w, "h": h}
            # Large camera scenes or non-faces: Return None to avoid out-of-distribution hallucinations
            return None, None

        # Pick largest detected face
        detected_faces = sorted(detected_faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = detected_faces[0]

        # Apply 1.20x margin with boundary clamping
        margin_x = int(fw * 0.18)
        margin_y = int(fh * 0.18)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)

        face_crop = image_bgr[y1:y2, x1:x2]
        bbox = {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}
        return face_crop, bbox

    def derive_verdict(self, fake_conf: float) -> str:
        """Translates probability into 3-tier qualitative forensic verdict with calibrated thresholds."""
        if fake_conf < 0.48:
            return "AUTHENTIC"
        elif fake_conf < 0.68:
            return "SUSPICIOUS"
        else:
            return "LIKELY_FAKE"

    def analyze_image(self, image_bytes: bytes, original_filename: str, upload_dir: str) -> dict:
        if not self.is_ready:
            raise RuntimeError("DeepfakeInferenceService is not ready. Models not loaded.")

        # 1. Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Corrupted or unreadable image file; OpenCV pixel decoding failed.")

        # 2. Setup upload directories
        images_dir = os.path.join(upload_dir, "deepfake", "images")
        heatmaps_dir = os.path.join(upload_dir, "deepfake", "heatmaps")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(heatmaps_dir, exist_ok=True)

        file_uuid = uuid.uuid4().hex[:12]
        clean_ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
        stored_filename = f"df_{file_uuid}{clean_ext}"
        heatmap_filename = f"df_{file_uuid}_heatmap.png"

        image_disk_path = os.path.join(images_dir, stored_filename)
        heatmap_disk_path = os.path.join(heatmaps_dir, heatmap_filename)

        # Save uploaded image
        cv2.imwrite(image_disk_path, image_bgr)

        # 3. Face localization
        face_crop, face_bbox = self.detect_face(image_bgr)

        if face_crop is None:
            # Graceful degradation: No face detected in the image
            cv_features = extract_features(image_bgr)
            return {
                "filename": original_filename,
                "stored_filename": stored_filename,
                "fake_confidence": 0.0,
                "verdict": "NOT_DETECTED",
                "face_detected": False,
                "face_bbox": None,
                "image_url": f"/uploads/deepfake/images/{stored_filename}",
                "heatmap_url": None,
                "statistics": {
                    "note": "No frontal human face localized. Please upload a clear portrait.",
                    "calibrated_threshold": round(float(self.calibrated_threshold), 3),
                    "all_features": {k: round(float(v), 4) for k, v in cv_features.items()}
                }
            }

        # 4. Neural inference
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transforms(image=rgb_face)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            fake_conf = float(torch.sigmoid(logits).item())

        verdict = self.derive_verdict(fake_conf)

        # 5. Grad-CAM explainability heatmap
        cam_map = self.gradcam.generate_cam(tensor)
        crop_overlay, crop_raw_heatmap = self.gradcam.generate_overlay(face_crop, cam_map, alpha=0.35)

        # Stitch overlay onto full image if cropped
        if face_bbox and face_bbox["w"] != image_bgr.shape[1]:
            full_overlay = image_bgr.copy()
            x, y, w, h = face_bbox["x"], face_bbox["y"], face_bbox["w"], face_bbox["h"]
            full_overlay[y:y+h, x:x+w] = crop_overlay
            cv2.imwrite(heatmap_disk_path, full_overlay)
        else:
            cv2.imwrite(heatmap_disk_path, crop_overlay)

        # 6. Spectral & deterministic forensic statistics (all 22 CV features + forensic metadata)
        cv_features = extract_features(image_bgr)
        stats = {
            "fft_high_freq_ratio": round(float(cv_features.get("fft_high_freq_ratio", 0.0)), 4),
            "dct_blockiness": round(float(cv_features.get("dct_blockiness", 0.0)), 2),
            "laplacian_variance": round(float(cv_features.get("laplacian_variance", 0.0)), 2),
            "noise_sigma": round(float(cv_features.get("noise_sigma_immerkaar", 0.0)), 3),
            "noise_sigma_immerkaar": round(float(cv_features.get("noise_sigma_immerkaar", 0.0)), 3),
            "mean_luminance": round(float(cv_features.get("mean_luminance", 0.0)), 2),
            "rms_contrast": round(float(cv_features.get("rms_contrast", 0.0)), 3),
            "mean_saturation": round(float(cv_features.get("mean_saturation", 0.0)), 3),
            "glcm_contrast": round(float(cv_features.get("glcm_contrast", 0.0)), 2),
            "calibrated_threshold": round(float(self.calibrated_threshold), 3),
            "face_width_px": face_bbox["w"] if face_bbox else None,
            "face_height_px": face_bbox["h"] if face_bbox else None,
            "face_x": face_bbox["x"] if face_bbox else None,
            "face_y": face_bbox["y"] if face_bbox else None,
            "all_features": {k: round(float(v), 4) for k, v in cv_features.items()}
        }

        return {
            "filename": original_filename,
            "stored_filename": stored_filename,
            "fake_confidence": round(fake_conf, 4),
            "verdict": verdict,
            "face_detected": True,
            "face_bbox": face_bbox,
            "image_url": f"/uploads/deepfake/images/{stored_filename}",
            "heatmap_url": f"/uploads/deepfake/heatmaps/{heatmap_filename}",
            "statistics": stats
        }

deepfake_service = DeepfakeInferenceService()
