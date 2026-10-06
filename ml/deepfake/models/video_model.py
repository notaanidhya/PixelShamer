"""
ml/deepfake/models/video_model.py
=================================
Spatio-Temporal Deepfake Video Detection Architecture.
Combines:
  1. Spatial Feature Extractor (Phase 1 EfficientNet-B5, 2048-dim embeddings)
  2. Temporal Sequence Aggregator (Bidirectional LSTM, 2-layer, hidden=256)
  3. Temporal Self-Attention Mechanism (Weights peak anomalous frames higher)
  4. Dual-Head Classification: Video-level verdict + Frame-by-frame timeline
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from ml.deepfake.models.efficientnet_deepfake import build_model

class TemporalAttention(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C)
        Returns:
            context: (B, C) attention-weighted pooled vector
            weights: (B, T) normalized frame attention probabilities
        """
        scores = self.net(x) # (B, T, 1)
        weights = torch.softmax(scores, dim=1) # (B, T, 1)
        context = torch.sum(weights * x, dim=1) # (B, C)
        return context, weights.squeeze(-1)

class DeepfakeVideoModel(nn.Module):
    def __init__(
        self,
        spatial_checkpoint: str | None = None,
        spatial_backbone: str = "efficientnet_b5",
        spatial_feature_dim: int | None = None,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        freeze_spatial: bool = True
    ):
        super().__init__()
        
        # Auto-detect architecture from checkpoint if present
        ckpt_state = None
        if spatial_checkpoint and os.path.exists(spatial_checkpoint):
            ckpt = torch.load(spatial_checkpoint, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict):
                spatial_backbone = ckpt.get("architecture", spatial_backbone)
                ckpt_state = ckpt.get("model_state", ckpt)
            else:
                ckpt_state = ckpt
            print(f"[*] Loaded spatial checkpoint ({spatial_backbone}) from: {spatial_checkpoint}")

        if spatial_feature_dim is None:
            spatial_feature_dim = 2048 if "b5" in spatial_backbone else 1408

        self.spatial_feature_dim = spatial_feature_dim
        self.spatial_backbone = spatial_backbone
        self.hidden_dim = hidden_dim
        self.freeze_spatial = freeze_spatial

        # 1. Spatial Backbone (EfficientNet-B5 or B2)
        self.spatial_cnn = build_model(
            model_name=spatial_backbone,
            pretrained=False,
            freeze_early=False
        )

        if ckpt_state is not None:
            self.spatial_cnn.load_state_dict(ckpt_state, strict=False)

        # Replace classification head on backbone to extract clean pooling features
        self.spatial_cnn.backbone.reset_classifier(0)

        if freeze_spatial:
            self.freeze_spatial_backbone()

        # 2. Temporal Sequence Modeling (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=spatial_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_lstm_layers > 1 else 0.0
        )

        lstm_out_dim = hidden_dim * 2 # Bidirectional

        # 3. Temporal Self-Attention Pooling
        self.attention = TemporalAttention(in_features=lstm_out_dim, hidden_dim=64)

        # 4. Final Video Classification Head (Clip Verdict)
        self.clip_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # 5. Frame-level Auxiliary Classifier (For Frame Timeline)
        self.frame_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(lstm_out_dim, 1)
        )

    def freeze_spatial_backbone(self):
        """Freezes all spatial CNN parameters to speed up temporal training."""
        for p in self.spatial_cnn.parameters():
            p.requires_grad = False
        self.freeze_spatial = True

    def unfreeze_spatial_backbone(self):
        """Unfreezes spatial CNN for end-to-end joint fine-tuning."""
        for p in self.spatial_cnn.parameters():
            p.requires_grad = True
        self.freeze_spatial = False

    def forward(
        self,
        x: torch.Tensor,
        is_pre_extracted: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass supporting both raw frame tensors and pre-extracted features.
        
        Args:
            x: Raw frames (B, T, 3, H, W) OR Pre-extracted features (B, T, spatial_feature_dim)
            is_pre_extracted: Set to True if x is already feature embeddings.
            
        Returns:
            clip_logits: (B, 1) Logit for overall video classification
            frame_logits: (B, T) Logits for individual frame predictions
            attention_weights: (B, T) Attention score distribution across frames
        """
        if is_pre_extracted:
            features = x # (B, T, 2048)
        else:
            b, t, c, h, w = x.shape
            # Flatten batch and sequence to run spatial CNN in parallel
            x_flat = x.view(b * t, c, h, w)
            if self.freeze_spatial:
                with torch.no_grad():
                    spatial_features = self.spatial_cnn.backbone.forward_features(x_flat)
                    pooled = self.spatial_cnn.backbone.forward_head(spatial_features, pre_logits=True)
            else:
                spatial_features = self.spatial_cnn.backbone.forward_features(x_flat)
                pooled = self.spatial_cnn.backbone.forward_head(spatial_features, pre_logits=True)

            features = pooled.view(b, t, -1) # (B, T, 2048)

        # Temporal sequence processing
        lstm_out, _ = self.lstm(features) # (B, T, hidden_dim * 2)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out) # (B, 512), (B, T)

        # Video clip prediction
        clip_logits = self.clip_classifier(context) # (B, 1)

        # Frame-by-frame anomaly predictions
        frame_logits = self.frame_classifier(lstm_out).squeeze(-1) # (B, T)

        return clip_logits, frame_logits, attn_weights

    def predict_video(
        self,
        x: torch.Tensor,
        calibrated_threshold: float = 0.50
    ) -> dict:
        """
        Inference helper returning human-interpretable results:
        - video_probability: overall probability of forgery [0, 1]
        - prediction: 'FAKE' or 'REAL'
        - frame_probabilities: list of per-frame probabilities
        - peak_anomalous_frame_idx: index of most suspicious frame
        """
        self.eval()
        with torch.no_grad():
            clip_logits, frame_logits, attn_weights = self.forward(x, is_pre_extracted=False)
            clip_prob = float(torch.sigmoid(clip_logits)[0, 0].cpu().numpy())
            frame_probs = torch.sigmoid(frame_logits)[0].cpu().numpy().tolist()
            attn = attn_weights[0].cpu().numpy().tolist()

            peak_frame_idx = int(np.argmax(frame_probs))

            return {
                "prediction": "FAKE" if clip_prob >= calibrated_threshold else "REAL",
                "video_probability": round(clip_prob, 4),
                "threshold": calibrated_threshold,
                "peak_anomalous_frame_idx": peak_frame_idx,
                "peak_anomalous_probability": round(frame_probs[peak_frame_idx], 4),
                "frame_probabilities": [round(p, 4) for p in frame_probs],
                "attention_weights": [round(a, 4) for a in attn]
            }

def build_video_model(
    spatial_checkpoint: str | None = None,
    spatial_backbone: str = "efficientnet_b5",
    hidden_dim: int = 256,
    freeze_spatial: bool = True
) -> DeepfakeVideoModel:
    dim = 2048 if "b5" in spatial_backbone else 1408
    return DeepfakeVideoModel(
        spatial_checkpoint=spatial_checkpoint,
        spatial_backbone=spatial_backbone,
        spatial_feature_dim=dim,
        hidden_dim=hidden_dim,
        freeze_spatial=freeze_spatial
    )
