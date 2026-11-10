"""3D U-Net training on a Medical Segmentation Decathlon task (default: Task09_Spleen) with MONAI.

  python -m medimg.train.train_segmentation --root data/msd --task Task09_Spleen --epochs 100
Downloads the task (~1.5 GB for Spleen) on first run; needs internet.
"""
import argparse
import json
import os

import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (AsDiscrete, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped,
                              LoadImaged, Orientationd, RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd)
from monai.utils import set_determinism

from medimg.models.segmentation import build_unet3d


def transforms(train: bool):
    keys = ["image", "label"]
    t = [
        LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes="RAS"),
        Spacingd(keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged("image", a_min=-57, a_max=175, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys, source_key="image"),
    ]
    if train:
        t.append(RandCropByPosNegLabeld(keys, label_key="label", spatial_size=(96, 96, 96),
                                        pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0))
    t.append(EnsureTyped(keys))
    return Compose(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/msd")
    ap.add_argument("--task", default="Task09_Spleen")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="models/unet3d_spleen.pt")
    ap.add_argument("--metrics-out", default="reports/seg_metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    set_determinism(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(root_dir=a.root, task=a.task, download=True, seed=a.seed, val_frac=0.2, cache_rate=0.5, num_workers=4)
    tr = DecathlonDataset(section="training", transform=transforms(True), **common)
    va = DecathlonDataset(section="validation", transform=transforms(False), **common)
    tl = DataLoader(tr, batch_size=2, shuffle=True, num_workers=2)
    vl = DataLoader(va, batch_size=1, num_workers=2)

    model = build_unet3d().to(dev)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    metric = DiceMetric(include_background=False, reduction="mean")
    post_p, post_l = AsDiscrete(argmax=True, to_onehot=2), AsDiscrete(to_onehot=2)

    best, history = -1.0, []
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a.metrics_out) or ".", exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for b in tl:
            x, y = b["image"].to(dev), b["label"].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot, n = tot + loss.item(), n + 1
        rec = {"epoch": ep + 1, "train_loss": tot / max(n, 1)}
        if (ep + 1) % a.val_every == 0:
            model.eval()
            with torch.no_grad():
                for b in vl:
                    x, y = b["image"].to(dev), b["label"].to(dev)
                    out = sliding_window_inference(x, (160, 160, 160), 4, model)
                    metric(y_pred=[post_p(o) for o in decollate_batch(out)], y=[post_l(l) for l in decollate_batch(y)])
                dice = metric.aggregate().item()
                metric.reset()
            rec["val_dice"] = dice
            if dice > best:
                best = dice
                torch.save({"state_dict": model.state_dict(), "task": a.task, "val_dice": dice}, a.out)
        history.append(rec)
        print(rec, flush=True)
    with open(a.metrics_out, "w") as f:
        json.dump({"task": a.task, "best_val_dice": best, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
