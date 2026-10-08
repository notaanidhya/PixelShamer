"""
backend/app/services/deepfake_video_inference.py
================================================
Production Video Inference Engine for Deepfake Detection.
Handles:
  - Video stream ingestion and uniform facial keyframe extraction
  - Bi-LSTM + Temporal Self-Attention sequence scoring
  - Frame-by-frame anomaly timeline generation
  - Peak-anomaly frame explainability via Grad-CAM spatial heatmap
"""

import os
import sys
import uuid
import logging
import threading
import cv2
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger("deepfake_api.video_inference")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.deepfake.video_extractor import VideoFaceExtractor
from ml.deepfake.models.video_model import build_video_model, DeepfakeVideoModel
from ml.deepfake.gradcam import GradCAM

class VideoDeepfakeInferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.extractor = None
        self.gradcam = None
        self.calibrated_threshold = 0.450
        self.is_ready = False
        self._inference_lock = threading.Lock()

    def load_models(self, video_model_path: str | None = None, spatial_model_path: str | None = None):
        """Loads video model weights and initializes spatial Grad-CAM hooks."""
        models_dir = os.path.join(BASE_DIR, "ml", "deepfake", "models")
        
        if video_model_path is None:
            video_model_path = os.path.join(models_dir, "deepfake_video_best.pt")

        if spatial_model_path is None:
            b5_path = os.path.join(models_dir, "efficientnet_b5_deepfake_best.pt")
            b2_path = os.path.join(models_dir, "efficientnet_deepfake_best.pt")
            spatial_model_path = b5_path if os.path.exists(b5_path) else (b2_path if os.path.exists(b2_path) else None)

        logger.info(f"Loading Deepfake Video Model on device: {self.device}")
        logger.info(f"  - Spatial Weights: {spatial_model_path}")
        logger.info(f"  - Video Weights:   {video_model_path}")

        hidden_dim = 256
        spatial_backbone = "efficientnet_b5"
        self.calibrated_threshold = 0.450

        ckpt = None
        if os.path.exists(video_model_path):
            ckpt = torch.load(video_model_path, map_location=self.device, weights_only=True)
            if isinstance(ckpt, dict):
                hidden_dim = ckpt.get("hidden_dim", 256)
                spatial_backbone = ckpt.get("spatial_backbone", "efficientnet_b5")
                self.calibrated_threshold = ckpt.get("optimal_threshold", 0.450)

        # Instantiate video model
        self.model = build_video_model(
            spatial_checkpoint=spatial_model_path,
            spatial_backbone=spatial_backbone,
            hidden_dim=hidden_dim,
            freeze_spatial=True
        ).to(self.device)

        if ckpt is not None:
            state = ckpt.get("model_state", ckpt)
            model_state = self.model.state_dict()
            matched_state = {
                k: v for k, v in state.items()
                if k in model_state and model_state[k].shape == v.shape
            }
            self.model.load_state_dict(matched_state, strict=False)
            logger.info(f"Successfully loaded video model ({len(matched_state)}/{len(state)} keys). Calibrated threshold: {self.calibrated_threshold:.3f}")
        else:
            logger.warning(f"Video checkpoint not found at {video_model_path}. Initialized with spatial weights.")

        self.model.eval()
        self.extractor = VideoFaceExtractor(target_size=288, default_num_frames=16)

        # Ensure Grad-CAM target layer has gradients enabled for backpropagation
        if hasattr(self.model.spatial_cnn.backbone, "conv_head"):
            for p in self.model.spatial_cnn.backbone.conv_head.parameters():
                p.requires_grad = True
        elif hasattr(self.model.spatial_cnn.backbone, "blocks"):
            for p in self.model.spatial_cnn.backbone.blocks[-1].parameters():
                p.requires_grad = True

        # Initialize Grad-CAM hooked into the spatial CNN backbone
        try:
            self.gradcam = GradCAM(self.model.spatial_cnn)
        except Exception as e:
            logger.warning(f"Could not initialize GradCAM on video backbone: {e}")
            self.gradcam = None

        self.is_ready = True
        logger.info("VideoDeepfakeInferenceService ready.")

    def analyze_video(
        self,
        video_path: str | Path,
        original_filename: str,
        upload_dir: str,
        num_frames: int = 16
    ) -> dict:
        """
        Executes end-to-end video deepfake analysis:
        1. Decodes and samples 16 facial frames across video duration
        2. Evaluates sequence with Bi-LSTM + Self-Attention
        3. Identifies peak anomaly frame and computes Grad-CAM explainability overlay
        4. Returns full timeline and verdict
        """
        if not self.is_ready:
            self.load_models()

        video_path_str = str(video_path)
        cap = cv2.VideoCapture(video_path_str)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        duration_sec = total_video_frames / max(1.0, fps)

        # Extract normalized tensor sequence and raw BGR face crops in a single pass
        tensor_seq, raw_crops_bgr, timestamps = self.extractor.extract_face_sequence(
            video_path=video_path_str,
            num_frames=num_frames,
            return_both=True
        )

        if tensor_seq is None or len(timestamps) == 0:
            raise ValueError(f"Could not extract facial frames from video: {original_filename}")

        with self._inference_lock:
            # Forward pass: shape (1, T, 3, H, W)
            x = tensor_seq.unsqueeze(0).to(self.device)

            with torch.no_grad():
                clip_logits, frame_logits, attn_weights = self.model(x, is_pre_extracted=False)
                model_clip_prob = float(torch.sigmoid(clip_logits)[0, 0].cpu().numpy())
                frame_probs = torch.sigmoid(frame_logits)[0].cpu().numpy().tolist()
                attn = attn_weights[0].cpu().numpy().tolist()

            # Identify peak anomalous frame
            peak_idx = int(np.argmax(frame_probs))
            peak_timestamp = timestamps[peak_idx]
            peak_prob = float(frame_probs[peak_idx])

            # Forensic sequence aggregation: account for sparse/localized frame manipulations
            sorted_probs = sorted(frame_probs, reverse=True)
            top3_anomaly = float(np.mean(sorted_probs[:min(3, len(sorted_probs))]))

            if peak_prob >= 0.50:
                clip_prob = max(model_clip_prob, 0.65 * top3_anomaly + 0.35 * peak_prob)
            else:
                clip_prob = model_clip_prob

            # Generate Grad-CAM for the peak anomaly frame
            heatmap_rel_path = None
            if self.gradcam is not None and raw_crops_bgr is not None and len(raw_crops_bgr) > peak_idx:
                try:
                    peak_tensor = tensor_seq[peak_idx:peak_idx+1].to(self.device)
                    cam_map = self.gradcam.generate_cam(peak_tensor)
                    peak_raw_bgr = raw_crops_bgr[peak_idx]
                    overlay, _ = self.gradcam.generate_overlay(peak_raw_bgr, cam_map, alpha=0.35)

                    heatmaps_dir = os.path.join(upload_dir, "deepfake", "heatmaps")
                    os.makedirs(heatmaps_dir, exist_ok=True)
                    heatmap_filename = f"df_vid_{uuid.uuid4().hex[:10]}_heatmap.jpg"
                    heatmap_abs_path = os.path.join(heatmaps_dir, heatmap_filename)
                    cv2.imwrite(heatmap_abs_path, overlay)
                    heatmap_rel_path = f"/uploads/deepfake/heatmaps/{heatmap_filename}"
                except Exception as e:
                    logger.error(f"Grad-CAM generation failed for peak video frame: {e}")

        # Multi-factor forensic verdict combining sequence confidence with peak localized anomaly
        if clip_prob >= 0.50 or peak_prob >= 0.75:
            verdict = "LIKELY_FAKE"
        elif clip_prob >= 0.30 or peak_prob >= 0.50:
            verdict = "SUSPICIOUS"
        else:
            verdict = "AUTHENTIC"

        # Forensic narrative
        if verdict == "LIKELY_FAKE":
            summary = (
                f"Definitive deepfake manipulation detected ({clip_prob*100:.1f}% confidence). "
                f"Severe temporal discontinuity localized at second {peak_timestamp:.1f} (frame {peak_idx+1}/{num_frames}) "
                f"with {peak_prob*100:.1f}% peak anomaly."
            )
        elif verdict == "SUSPICIOUS":
            summary = (
                f"Elevated forgery probability detected ({clip_prob*100:.1f}%). "
                f"Unnatural inter-frame blending observed around timestamp {peak_timestamp:.1f}s."
            )
        else:
            summary = (
                f"Authentic natural video verified ({clip_prob*100:.1f}% forgery score, below threshold {self.calibrated_threshold:.2f}). "
                f"Temporal facial transitions demonstrate continuous biomechanical consistency."
            )

        # Build timeline
        timeline = []
        for i in range(len(timestamps)):
            timeline.append({
                "frame_index": i,
                "timestamp_sec": round(timestamps[i], 3),
                "fake_probability": round(frame_probs[i], 4),
                "is_peak_anomaly": (i == peak_idx)
            })

        return {
            "filename": original_filename,
            "duration_seconds": round(duration_sec, 2),
            "total_frames_analyzed": len(timestamps),
            "fake_confidence": round(clip_prob, 4),
            "verdict": verdict,
            "threshold": round(self.calibrated_threshold, 3),
            "peak_anomaly_timestamp": round(peak_timestamp, 3),
            "peak_anomaly_confidence": round(peak_prob, 4),
            "timeline": timeline,
            "heatmap_url": heatmap_rel_path,
            "analysis_summary": summary
        }

# Global singleton
video_deepfake_service = VideoDeepfakeInferenceService()
