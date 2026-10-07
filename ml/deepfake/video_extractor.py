"""
ml/deepfake/video_extractor.py
==============================
High-throughput video frame extraction and facial sequence tracking.
Extracts uniform keyframe sequences and applies smooth face tracking
with 1.25x expansion to preserve facial boundaries, blending margins,
and artifact perimeters for temporal forensic analysis.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from albumentations.pytorch import ToTensorV2
import albumentations as A

# ImageNet normalization for PyTorch backbones
FRAME_TRANSFORMS = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

class VideoFaceExtractor:
    def __init__(self, target_size: int = 288, default_num_frames: int = 16):
        self.target_size = target_size
        self.default_num_frames = default_num_frames
        self.face_cascade = None
        self.profile_cascade = None
        self._load_face_detectors()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["face_cascade"] = None
        state["profile_cascade"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_face_detectors()

    def _load_face_detectors(self):
        """Loads Haar cascades with multi-path resolution."""
        base_dir = Path(__file__).resolve().parent
        bundled_dir = base_dir / "models" / "haarcascades"
        sys_haarcascades = getattr(cv2.data, "haarcascades", "") if hasattr(cv2, "data") else ""
        sys_dir = Path(sys_haarcascades) if sys_haarcascades else bundled_dir

        frontal_candidates = [
            bundled_dir / "haarcascade_frontalface_default.xml",
            sys_dir / "haarcascade_frontalface_default.xml"
        ]
        profile_candidates = [
            bundled_dir / "haarcascade_profileface.xml",
            sys_dir / "haarcascade_profileface.xml"
        ]

        for p in frontal_candidates:
            if p.exists():
                self.face_cascade = cv2.CascadeClassifier(str(p))
                break

        for p in profile_candidates:
            if p.exists():
                self.profile_cascade = cv2.CascadeClassifier(str(p))
                break

    def detect_face_bbox(self, image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Detects primary face bounding box with smooth fallback:
        Returns (x, y, w, h) with 1.25x expanded margin, or None if no face found.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape

        faces = []
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )

        # Fallback to profile face if frontal missed
        if len(faces) == 0 and self.profile_cascade is not None and not self.profile_cascade.empty():
            faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )
            if len(faces) == 0:
                flipped = cv2.flip(gray, 1)
                flipped_faces = self.profile_cascade.detectMultiScale(
                    flipped, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
                )
                if len(flipped_faces) > 0:
                    fx, fy, fw, fh = max(flipped_faces, key=lambda b: b[2] * b[3])
                    faces = [(w_img - (fx + fw), fy, fw, fh)]

        if len(faces) == 0:
            return None

        # Select largest face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

        # 1.25x margin expansion
        margin_x = int(w * 0.125)
        margin_y = int(h * 0.125)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        return (x1, y1, x2 - x1, y2 - y1)

    def extract_face_sequence(
        self,
        video_path: str | Path,
        num_frames: int | None = None,
        return_tensors: bool = True
    ) -> tuple[torch.Tensor | np.ndarray | None, list[float]]:
        """
        Extracts uniformly sampled facial crops from a video.
        
        Args:
            video_path: Path to .mp4 / .avi / .mov video file.
            num_frames: Number of evenly spaced frames to extract (default: 16).
            return_tensors: If True, returns torch.Tensor of shape (T, 3, target_size, target_size).
                           If False, returns np.ndarray of shape (T, target_size, target_size, 3) in BGR.
        
        Returns:
            frames: Sequence of face crops (T, 3, H, W) or (T, H, W, 3).
            timestamps: List of frame timestamps in seconds.
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        num_frames = num_frames or self.default_num_frames
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None, []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total_frames <= 0:
            cap.release()
            return None, []

        # Uniform keyframe index selection
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
            # Pad with last frame if video is shorter than num_frames
            while len(frame_indices) < num_frames:
                frame_indices.append(total_frames - 1)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int).tolist()

        face_crops = []
        timestamps = []
        last_valid_bbox = None

        current_frame_idx = 0
        target_pos = 0

        while cap.isOpened() and target_pos < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx == frame_indices[target_pos]:
                h_img, w_img = frame.shape[:2]
                bbox = self.detect_face_bbox(frame)

                # Smooth tracking: if face detector misses in this frame, use last known good bbox
                if bbox is not None:
                    last_valid_bbox = bbox
                elif last_valid_bbox is not None:
                    bbox = last_valid_bbox
                else:
                    # Fallback to center square crop if face not yet detected
                    min_dim = min(h_img, w_img)
                    x1 = (w_img - min_dim) // 2
                    y1 = (h_img - min_dim) // 2
                    bbox = (x1, y1, min_dim, min_dim)

                x, y, w, h = bbox
                crop = frame[y:y+h, x:x+w]
                crop_resized = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                face_crops.append(crop_resized)
                timestamps.append(round(current_frame_idx / fps, 3))
                target_pos += 1

            current_frame_idx += 1

        cap.release()

        # If video ended early, pad with the last frame
        while len(face_crops) < num_frames and len(face_crops) > 0:
            face_crops.append(face_crops[-1])
            timestamps.append(timestamps[-1])

        if len(face_crops) == 0:
            return None, []

        face_crops_np = np.stack(face_crops, axis=0) # (T, H, W, 3) in BGR

        if not return_tensors:
            return face_crops_np, timestamps

        # Convert to PyTorch Tensor: (T, 3, H, W) normalized
        tensors = []
        for img_bgr in face_crops:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            transformed = FRAME_TRANSFORMS(image=img_rgb)["image"]
            tensors.append(transformed)

        tensor_seq = torch.stack(tensors, dim=0) # (T, 3, H, W)
        return tensor_seq, timestamps
