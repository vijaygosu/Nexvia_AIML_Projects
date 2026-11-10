"""Inference engine: DICOM pixels -> multi-label probabilities + Grad-CAM heatmap."""
import io
import os

import numpy as np
import torch
from PIL import Image

from medimg.data.transforms import eval_transform
from medimg.explain.gradcam import GradCAM, cam_to_rgba
from medimg.models.classifier import build_untrained, load_checkpoint

MAX_WEB_SIDE = 1024


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class InferenceEngine:
    def __init__(self, ckpt_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if ckpt_path and os.path.exists(ckpt_path):
            self.model, ck = load_checkpoint(ckpt_path, self.device)
            self.trained = True
        else:
            self.model, ck = build_untrained()
            self.trained = False
        self.model.to(self.device).eval()
        self.labels, self.version, self.img_size = ck["labels"], ck["version"], ck["img_size"]
        self.metrics = ck.get("metrics", {})
        self.tf = eval_transform(self.img_size)
        self.cam = GradCAM(self.model)

    def _prep(self, gray_uint8: np.ndarray):
        return self.tf(Image.fromarray(gray_uint8)).unsqueeze(0).to(self.device)

    @staticmethod
    def _web_image(gray_uint8: np.ndarray) -> Image.Image:
        im = Image.fromarray(gray_uint8)
        scale = MAX_WEB_SIDE / max(im.size)
        return im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR) if scale < 1 else im

    def heatmap_png(self, gray_uint8: np.ndarray, label: str) -> bytes:
        cam, _, _ = self.cam(self._prep(gray_uint8), class_idx=self.labels.index(label))
        base = self._web_image(gray_uint8)
        c = Image.fromarray((cam[0].numpy() * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
        return _png(cam_to_rgba(np.asarray(c, dtype=np.float32) / 255))

    def predict(self, gray_uint8: np.ndarray) -> dict:
        x = self._prep(gray_uint8)
        _, logits, idx = self.cam(x)  # default: Grad-CAM for the top-scoring class
        probs = torch.sigmoid(logits)[0].numpy()
        top = int(probs.argmax())
        preds = {l: round(float(p), 4) for l, p in zip(self.labels, probs)}
        top_label = self.labels[top]
        return {
            "predictions": preds, "top_label": top_label, "top_prob": preds[top_label],
            "model_version": self.version,
            "image_png": _png(np.asarray(self._web_image(gray_uint8))),
            "heatmap_png": self.heatmap_png(gray_uint8, top_label),
        }
