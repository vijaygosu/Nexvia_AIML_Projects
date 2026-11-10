"""Transfer-learning training of DenseNet121 on the prepared manifest.

  python -m medimg.train.train_classifier --manifest data/processed/manifest.csv \
      --out models/densenet121_nih.pt --epochs 8 --batch-size 32
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from medimg.config import LABELS_8, LABELS_14
from medimg.data.dataset import ChestXrayDataset
from medimg.data.transforms import eval_transform, train_transform
from medimg.models.classifier import ChestClassifier, save_checkpoint


def mean_auroc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y, _ in loader:
        ps.append(torch.sigmoid(model(x.to(device))).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--labels", type=int, choices=(8, 14), default=8)
    ap.add_argument("--out", default="models/densenet121_nih.pt")
    ap.add_argument("--metrics-out", default="reports/train_metrics.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--version", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    labels = LABELS_8 if a.labels == 8 else LABELS_14
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = ChestXrayDataset(a.manifest, "train", labels, train_transform())
    va = ChestXrayDataset(a.manifest, "val", labels, eval_transform())
    tl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=dev == "cuda", drop_last=True)
    vl = DataLoader(va, a.batch_size * 2, num_workers=a.workers)

    prev = tr.targets().mean(0).clamp(min=1e-4)
    pos_weight = ((1 - prev) / prev).clamp(max=30).to(dev)  # class imbalance
    model = ChestClassifier(len(labels), pretrained=True).to(dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    version = a.version or time.strftime("densenet121-%Y%m%d-%H%M")

    history, best = [], -1.0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a.metrics_out) or ".", exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        run = 0.0
        for i, (x, y, _) in enumerate(tl):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        yv, pv = predict(model, vl, dev)
        auc = mean_auroc(yv, pv)
        history.append({"epoch": ep + 1, "train_loss": run / max(i + 1, 1), "val_mean_auroc": auc})
        print(history[-1], flush=True)
        if auc > best:
            best = auc
            save_checkpoint(a.out, model, labels, version, {"val_mean_auroc": auc, "epoch": ep + 1})
    with open(a.metrics_out, "w") as f:
        json.dump({"version": version, "best_val_mean_auroc": best, "history": history}, f, indent=2)
    print("saved", a.out, "best val mean AUROC", round(best, 4))


if __name__ == "__main__":
    main()
