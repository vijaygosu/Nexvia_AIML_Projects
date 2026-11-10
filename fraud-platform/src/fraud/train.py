"""Train XGBoost (+ GNN), blend, pick a threshold, evaluate on a held-out future window,
log everything to MLflow and write the serving artifacts.

    python -m fraud.train --data synthetic
    python -m fraud.train --data ieee --ieee-dir /path/to/ieee-fraud-detection
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os

import numpy as np
import pandas as pd

from .data import CANON, load_ieee, make_synthetic, split_times
from .features import FEATURES, build_features, replay_states
from .graph import compute_user_scores
from .metrics import best_f1_threshold, evaluate, pr_auc
from .model_io import save_model

try:
    import mlflow
except ImportError:                                           # tracking becomes a no-op
    mlflow = None


@contextlib.contextmanager
def tracking(experiment, run_name):
    if mlflow is None:
        print("[warn] mlflow not installed - metrics only written to disk")
        yield None
        return
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield mlflow


def fit_model(kind, Xtr, ytr, Xva, yva, imbalance, seed):
    pos = max(int(ytr.sum()), 1)
    spw = (len(ytr) - pos) / pos
    if imbalance == "smote":
        from imblearn.over_sampling import SMOTE
        Xtr, ytr = SMOTE(sampling_strategy=0.1, random_state=seed).fit_resample(Xtr, ytr)
        spw = 1.0
    if kind == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                          eval_metric="aucpr", early_stopping_rounds=40, n_jobs=-1, random_state=seed)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m, {"scale_pos_weight": spw}
    from sklearn.ensemble import HistGradientBoostingClassifier   # sandbox/smoke-test stand-in only
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=seed)
    return m.fit(Xtr, ytr), {"class_weight": "balanced"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "ieee"], default="synthetic")
    ap.add_argument("--ieee-dir")
    ap.add_argument("--nrows", type=int)
    ap.add_argument("--n-txn", type=int, default=150_000)
    ap.add_argument("--model", choices=["xgboost", "sklearn"], default="xgboost")
    ap.add_argument("--imbalance", choices=["class_weight", "smote"], default="class_weight")
    ap.add_argument("--gnn", choices=["auto", "pyg", "ring"], default="auto")
    ap.add_argument("--out", default="models")
    ap.add_argument("--results", default="docs/results")
    ap.add_argument("--experiment", default="fraud-detection")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if a.data == "synthetic":
        raw = make_synthetic(n_txn=a.n_txn, seed=a.seed)
    else:
        raw = load_ieee(os.path.join(a.ieee_dir, "train_transaction.csv"),
                        os.path.join(a.ieee_dir, "train_identity.csv"), a.nrows)
    df = build_features(raw)
    t1, t2 = split_times(df)
    tr, va, te = df.ts < t1, (df.ts >= t1) & (df.ts < t2), df.ts >= t2
    print(f"rows={len(df):,} fraud={df.is_fraud.mean():.3%} | train={tr.sum():,} val={va.sum():,} test={te.sum():,}")

    X, y = df[FEATURES], df["is_fraud"].to_numpy()
    with tracking(a.experiment, a.run_name) as mlf:
        model, extra = fit_model(a.model, X[tr], y[tr], X[va], y[va], a.imbalance, a.seed)
        px = model.predict_proba(X)[:, 1]

        scores, gnn_backend = compute_user_scores(df, t1, a.gnn, a.seed)
        pg = df["user_id"].map(scores).to_numpy(dtype=float)
        pg_eff = np.where(np.isnan(pg), px, pg)

        ws = np.linspace(0, 1, 21)
        w = float(ws[int(np.argmax([pr_auc(y[va], x * px[va] + (1 - x) * pg_eff[va]) for x in ws]))])
        pb = w * px + (1 - w) * pg_eff
        thr = best_f1_threshold(y[va], pb[va])

        res = {
            "dataset": a.data, "rows": int(len(df)), "gnn_backend": gnn_backend, "model": a.model,
            "imbalance": a.imbalance, "blend_w_xgb": w, "threshold": thr,
            "val": {"xgb_pr_auc": pr_auc(y[va], px[va]), "gnn_pr_auc": pr_auc(y[va], pg_eff[va]),
                    "blend_pr_auc": pr_auc(y[va], pb[va])},
            "test": {"xgb": evaluate(y[te], px[te], best_f1_threshold(y[va], px[va])),
                     "gnn": evaluate(y[te], pg_eff[te], best_f1_threshold(y[va], pg_eff[va])),
                     "blend": evaluate(y[te], pb[te], thr)},
        }
        print(json.dumps({k: res[k] for k in ("gnn_backend", "blend_w_xgb", "threshold", "val")}, indent=2))
        for name in ("xgb", "gnn", "blend"):
            r = res["test"][name]
            print(f"TEST {name:6s} PR-AUC={r['pr_auc']:.4f} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

        # ---- artifacts used by serving / drift / replay
        fname = save_model(model, a.out)
        scores.rename_axis("user_id").reset_index().to_csv(os.path.join(a.out, "gnn_scores.csv"), index=False)
        states = replay_states(df[df.ts < t2])                 # state as of the start of the test window
        pd.DataFrame([{"user_id": u, **s.to_dict(), "recent_ts": json.dumps(s.recent_ts)}
                      for u, s in states.items()]).to_csv(os.path.join(a.out, "user_state.csv"), index=False)
        meta = {"features": FEATURES, "model_file": fname, "threshold": thr, "blend_w": w,
                "gnn_backend": gnn_backend, "t_train_end": float(t1), "t_val_end": float(t2)}
        json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
        json.dump(res, open(os.path.join(a.results, "metrics.json"), "w"), indent=2)
        df.loc[te, CANON].to_csv("data/replay.csv", index=False)
        df.loc[tr, FEATURES].sample(min(20_000, int(tr.sum())), random_state=a.seed).to_csv(
            "data/reference_features.csv", index=False)
        _plots(a.results, y[te], {"XGBoost": px[te], "Graph": pg_eff[te], "Blend": pb[te]}, model, FEATURES)

        if mlf:
            mlf.log_params({"data": a.data, "model": a.model, "imbalance": a.imbalance,
                            "gnn_backend": gnn_backend, "blend_w": w, "threshold": thr, **extra})
            mlf.log_metrics({"val_blend_pr_auc": res["val"]["blend_pr_auc"],
                             **{f"test_{n}_pr_auc": res["test"][n]["pr_auc"] for n in ("xgb", "gnn", "blend")},
                             "test_blend_precision": res["test"]["blend"]["precision"],
                             "test_blend_recall": res["test"]["blend"]["recall"]})
            mlf.log_artifacts(a.results)
            mlf.log_artifact(os.path.join(a.out, "meta.json"))
            if a.model == "xgboost":
                import mlflow.xgboost
                mlflow.xgboost.log_model(model, "model")
    return res


def _plots(out, y, curves, model, feats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n, p in curves.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{n} (AP={pr_auc(y, p):.3f})")
    ax.axhline(y.mean(), ls=":", c="gray", label="no skill")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (held-out future window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pr_curve.png"), dpi=140)
    plt.close(fig)
    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        order = np.argsort(imp)
        ax.barh(np.array(feats)[order], np.asarray(imp)[order])
        ax.set_title("Feature importance")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "feature_importance.png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
