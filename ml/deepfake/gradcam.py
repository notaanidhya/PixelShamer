"""
ml/deepfake/gradcam.py
======================
Gradient-weighted Class Activation Mapping (Grad-CAM) for EfficientNet-B5.
Produces spatial explainability heatmaps highlighting face forgery regions.
"""

import cv2
import torch
import numpy as np
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer=None):
        """
        Initializes Grad-CAM hook on the target layer of EfficientNet-B2.
        Default target layer: model.backbone.conv_head (terminal 1x1 conv before pooling).
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model.backbone, "conv_head"):
                self.target_layer = model.backbone.conv_head
            else:
                self.target_layer = model.backbone.blocks[-1]
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate_cam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM map in [0, 1].
        input_tensor: shape (1, 3, H, W)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor) # (1, 1)

        # Target fake class activation
        score = logits[0, 0]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True) # (1, C, 1, 1)

        # Weighted combination of activation maps: ReLU(sum(alpha_k * A_k))
        cam = F.relu((alpha * self.activations).sum(dim=1, keepdim=True)) # (1, 1, H', W')
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def generate_overlay(
        self,
        original_bgr: np.ndarray,
        cam_map: np.ndarray,
        alpha: float = 0.35
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates:
        1. overlay: Original image with 35% JET colormap blend
        2. raw_heatmap: Unblended JET colormap upsampled to original dimensions
        """
        h, w = original_bgr.shape[:2]
        resized_cam = cv2.resize(cam_map, (w, h), interpolation=cv2.INTER_CUBIC)
        norm_cam = np.clip(resized_cam * 255.0, 0, 255).astype(np.uint8)

        # JET colormap: Red = Highest Forgery Evidence, Blue = Nominal
        color_heatmap = cv2.applyColorMap(norm_cam, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_bgr, 1.0 - alpha, color_heatmap, alpha, 0)

        return overlay, color_heatmap
