"""
ml/deepfake/models/efficientnet_deepfake.py
===========================================
EfficientNet-B2 Binary Deepfake Classifier with Grad-CAM activation tapping.
Optimized for high-accuracy face forgery detection with low compute footprint.
"""

import torch
import torch.nn as nn
import timm

class EfficientNetDeepfake(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_early_blocks: bool = True):
        super().__init__()
        # Load EfficientNet-B2 backbone
        self.backbone = timm.create_model("efficientnet_b2", pretrained=pretrained)
        in_features = self.backbone.classifier.in_features # 1408
        
        # Replace classifier with regularized forensic head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )
        
        # Storage for Grad-CAM feature map and gradient
        self._target_activations = None
        self._target_gradients = None
        self._hook_handles = []

        if freeze_early_blocks:
            self.freeze_stages(freeze_up_to_stage=5)

    def freeze_stages(self, freeze_up_to_stage: int = 5):
        """Freezes earlier MBConv stages for fast transfer learning and anti-overfitting."""
        # Conv stem always frozen initially
        for p in self.backbone.conv_stem.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, "bn1") and self.backbone.bn1 is not None:
            for p in self.backbone.bn1.parameters():
                p.requires_grad = False
                
        # Freeze blocks up to stage index
        for i, block in enumerate(self.backbone.blocks):
            if i < freeze_up_to_stage:
                for p in block.parameters():
                    p.requires_grad = False
            else:
                for p in block.parameters():
                    p.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all parameters for fine-grained end-to-end tuning."""
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass.
        Returns:
            logits: (B, 1) raw unscaled logits (for BCEWithLogitsLoss)
            features: Optional last feature map tensor for Grad-CAM
        """
        # Forward features to tap the last MBConv block
        features = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(features, pre_logits=True)
        logits = self.backbone.classifier(pooled)

        if return_features:
            return logits, features
        return logits

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Returns calibrated sigmoid probability in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

def build_model(pretrained: bool = True, freeze_early: bool = True) -> EfficientNetDeepfake:
    return EfficientNetDeepfake(pretrained=pretrained, freeze_early_blocks=freeze_early)
