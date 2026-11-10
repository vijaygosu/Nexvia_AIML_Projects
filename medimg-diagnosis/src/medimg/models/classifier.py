"""DenseNet121 multi-label chest X-ray classifier with an explicit feature/head split (needed for Grad-CAM)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

from medimg.config import IMG_SIZE, LABELS_8


class ChestClassifier(nn.Module):
    def __init__(self, num_labels: int, pretrained: bool = True):
        super().__init__()
        net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = net.features
        self.classifier = nn.Linear(net.classifier.in_features, num_labels)

    def forward_features(self, x):
        # non-inplace ReLU so hooks/retain_grad on the returned tensor are safe
        return F.relu(self.features(x))

    def head(self, feats):
        return self.classifier(F.adaptive_avg_pool2d(feats, 1).flatten(1))

    def forward(self, x):
        return self.head(self.forward_features(x))


def save_checkpoint(path, model, labels, version, metrics=None, img_size=IMG_SIZE):
    torch.save({"state_dict": model.state_dict(), "labels": list(labels), "version": version,
                "img_size": img_size, "arch": "densenet121", "metrics": metrics or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = ChestClassifier(len(ck["labels"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def build_untrained(labels=None):
    """Random-weight fallback so the whole stack can be demoed without a trained checkpoint."""
    labels = list(labels or LABELS_8)
    return ChestClassifier(len(labels), pretrained=False).eval(), {
        "labels": labels, "version": "UNTRAINED-DEMO", "img_size": IMG_SIZE, "metrics": {},
    }
