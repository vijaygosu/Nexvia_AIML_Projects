import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)


def pr_auc(y, p):
    return float(average_precision_score(y, p)) if np.sum(y) > 0 else float("nan")


def best_f1_threshold(y, p) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[int(np.argmax(f1))])


def recall_at_precision(y, p, target=0.5) -> float:
    pr, rc, _ = precision_recall_curve(y, p)
    ok = rc[pr >= target]
    return float(ok.max()) if len(ok) else 0.0


def evaluate(y, p, thr) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pr_auc": pr_auc(y, p),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "threshold": float(thr), "precision": float(prec), "recall": float(rec),
        "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "recall_at_p50": recall_at_precision(y, p, 0.5),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fraud_rate": float(y.mean()),
    }
