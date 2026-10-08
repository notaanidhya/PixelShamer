"""
ml/deepfake/models/efficientnet_deepfake.py
===========================================
Binary Deepfake Classifier with Grad-CAM activation tapping.
Supports flexible backbones (EfficientNet-B5, B2, ConvNeXt) with dynamic feature extraction
and regularized forensic head.
"""

import torch
import torch.nn as nn
import timm

class EfficientNetDeepfake(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet_b5",
        pretrained: bool = True,
        freeze_early_blocks: bool = True,
        freeze_up_to_stage: int = 5
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Standard timm property for output feature channels across architectures
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        elif hasattr(self.backbone, "classifier") and hasattr(self.backbone.classifier, "in_features"):
            in_features = self.backbone.classifier.in_features
        else:
            in_features = 2048 if "b5" in model_name else 1408
            
        self.in_features = in_features

        # Replace classifier with regularized forensic head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )


        if freeze_early_blocks:
            self.freeze_stages(freeze_up_to_stage=freeze_up_to_stage)

    def freeze_stages(self, freeze_up_to_stage: int = 5):
        """Freezes earlier stages for fast transfer learning and anti-overfitting."""
        # Conv stem frozen initially if present
        if hasattr(self.backbone, "conv_stem"):
            for p in self.backbone.conv_stem.parameters():
                p.requires_grad = False
        if hasattr(self.backbone, "bn1") and self.backbone.bn1 is not None:
            for p in self.backbone.bn1.parameters():
                p.requires_grad = False

        # Freeze blocks up to stage index
        if hasattr(self.backbone, "blocks"):
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

def build_model(
    model_name: str = "efficientnet_b5",
    pretrained: bool = True,
    freeze_early: bool = True,
    freeze_up_to_stage: int = 5
) -> EfficientNetDeepfake:
    return EfficientNetDeepfake(
        model_name=model_name,
        pretrained=pretrained,
        freeze_early_blocks=freeze_early,
        freeze_up_to_stage=freeze_up_to_stage
    )

