"""Drift detection (Evidently + PSI fallback) and automatic retraining trigger."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .features import FEATURES

PSI_ALERT = 0.2
# user_txn_count only ever grows with time -> it "drifts" on a healthy stream; exclude it.
MONITORED = [f for f in FEATURES if f != "user_txn_count"]


def psi(ref, cur, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    eps = 1e-4
    uniq = np.unique(ref)
    if len(uniq) <= bins:                                  # discrete / binary features
        cats = np.union1d(uniq, np.unique(cur))
        r = np.array([(ref == c).mean() for c in cats])
        c = np.array([(cur == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def evidently_share(ref: pd.DataFrame, cur: pd.DataFrame, html_path: str):
    try:
        from evidently.report import Report                 # evidently==0.4.x API
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        return None
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref, current_data=cur)
    rep.save_html(html_path)
    return float(rep.as_dict()["metrics"][0]["result"]["share_of_drifted_columns"])


def run(ref: pd.DataFrame, cur: pd.DataFrame, out_dir="monitoring/reports",
        share_threshold=0.3, min_rows=500) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ref, cur = ref[MONITORED], cur[MONITORED]
    scores = {f: psi(ref[f], cur[f]) for f in MONITORED}
    drifted = sorted(f for f, v in scores.items() if v > PSI_ALERT)
    psi_share = len(drifted) / len(MONITORED)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ev_share = evidently_share(ref, cur, os.path.join(out_dir, f"drift_{stamp}.html")) if len(cur) >= 50 else None
    share = ev_share if ev_share is not None else psi_share
    result = {
        "time": stamp, "n_reference": len(ref), "n_current": len(cur),
        "psi": scores, "psi_drifted_features": drifted, "psi_share": psi_share,
        "evidently_share": ev_share, "share_used": share,
        "threshold": share_threshold, "enough_data": len(cur) >= min_rows,
        "retrain_triggered": bool(share >= share_threshold and len(cur) >= min_rows),
    }
    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_live_window(db_path: str, n: int = 5000) -> pd.DataFrame:
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(f"SELECT {','.join(FEATURES)} FROM scored ORDER BY txn_id DESC LIMIT {int(n)}", con)
    finally:
        con.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference_features.csv")
    ap.add_argument("--db", default=os.getenv("DB_PATH", "data/cases.db"))
    ap.add_argument("--current-csv", help="use a CSV instead of the live DB window")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--retrain", action="store_true", help="run the retrain command when triggered")
    ap.add_argument("--reload-url", help="POST here after retraining so the API hot-loads the new model")
    ap.add_argument("--retrain-cmd", default=f"{sys.executable} -m fraud.train --data synthetic --run-name drift-retrain")
    a = ap.parse_args(argv)
    ref = pd.read_csv(a.reference)
    cur = pd.read_csv(a.current_csv) if a.current_csv else load_live_window(a.db)
    res = run(ref, cur, out_dir=os.getenv("REPORT_DIR", "monitoring/reports"), share_threshold=a.threshold)
    print(json.dumps({k: res[k] for k in ("n_current", "psi_drifted_features", "share_used", "retrain_triggered")}, indent=2))
    if res["retrain_triggered"] and a.retrain:
        print("drift threshold exceeded -> retraining:", a.retrain_cmd)
        subprocess.run(a.retrain_cmd.split(), check=True)
        if a.reload_url:
            import requests
            print("reload:", requests.post(a.reload_url, timeout=30).text)
    return res


if __name__ == "__main__":
    main()
