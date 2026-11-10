"""Grad-CAM for ChestClassifier (Selvaraju et al., 2017)."""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        self.model = model

    def __call__(self, x: torch.Tensor, class_idx=None):
        """x: (B,3,H,W). Returns (cams (B,H,W) in [0,1], logits (B,C), class_idx (B,))."""
        self.model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # guarantees a graph even if params are frozen
            feats = self.model.forward_features(x)
            feats.retain_grad()
            logits = self.model.head(feats)
            if class_idx is None:
                idx = logits.argmax(1)
            elif isinstance(class_idx, int):
                idx = torch.full((x.shape[0],), class_idx, dtype=torch.long, device=x.device)
            else:
                idx = class_idx.to(x.device)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, idx[:, None]).sum().backward()
            weights = feats.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * feats).sum(1, keepdim=True)).detach()
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        mn, mx = flat.min(1).values[:, None, None], flat.max(1).values[:, None, None]
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.cpu(), logits.detach().cpu(), idx.cpu()


def jet(v: np.ndarray) -> np.ndarray:
    """Matplotlib-free 'jet' colormap. v in [0,1] -> (...,3) floats."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def cam_to_rgba(cam01: np.ndarray, max_alpha: float = 0.85) -> np.ndarray:
    """Heatmap as RGBA uint8; alpha grows with activation so low-evidence areas stay transparent."""
    rgb = jet(cam01)
    a = (cam01 * max_alpha)[..., None]
    return (np.concatenate([rgb, a], -1) * 255).astype(np.uint8)


def blend(gray_uint8: np.ndarray, cam01: np.ndarray, max_alpha: float = 0.85) -> np.ndarray:
    """Static composite (RGB uint8) for reports and figures."""
    rgba = cam_to_rgba(cam01, max_alpha).astype(np.float32) / 255
    base = np.repeat(gray_uint8[..., None], 3, -1).astype(np.float32) / 255
    out = base * (1 - rgba[..., 3:]) + rgba[..., :3] * rgba[..., 3:]
    return (out * 255).astype(np.uint8)
