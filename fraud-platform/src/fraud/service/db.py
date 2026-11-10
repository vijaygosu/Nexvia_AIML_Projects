"""SQLite store for analyst cases + the scored-feature log that feeds drift monitoring."""
import sqlite3
import threading
import time

import pandas as pd

from ..features import FEATURES


class DB:
    def __init__(self, path: str):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        cols = ", ".join(f"{f} REAL" for f in FEATURES)
        with self.lock, self.con:
            self.con.execute("""CREATE TABLE IF NOT EXISTS cases(
                case_id INTEGER PRIMARY KEY AUTOINCREMENT, txn_id INTEGER, user_id INTEGER,
                ts INTEGER, amount REAL, score REAL, xgb_score REAL, gnn_score REAL,
                status TEXT DEFAULT 'open', label TEXT, analyst TEXT, note TEXT,
                sim_label INTEGER, created_at REAL, reviewed_at REAL)""")
            self.con.execute(f"""CREATE TABLE IF NOT EXISTS scored(
                txn_id INTEGER, created_at REAL, score REAL, is_alert INTEGER, sim_label INTEGER,
                latency_ms REAL, e2e_ms REAL, {cols})""")

    def log(self, res: dict, sim_label=None, e2e_ms=None):
        f = res["features"]
        with self.lock, self.con:
            self.con.execute(
                f"INSERT INTO scored VALUES (?,?,?,?,?,?,?,{','.join('?' * len(FEATURES))})",
                (res["txn_id"], time.time(), res["fraud_score"], int(res["is_alert"]), sim_label,
                 res["latency_ms"], e2e_ms, *[f[c] for c in FEATURES]))
            if res["is_alert"]:
                self.con.execute(
                    "INSERT INTO cases(txn_id,user_id,ts,amount,score,xgb_score,gnn_score,sim_label,created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (res["txn_id"], res["user_id"], res["ts"], res["amount"], res["fraud_score"],
                     res["xgb_score"], res["gnn_score"], sim_label, time.time()))

    def alerts(self, status=None, limit=200):
        q, args = "SELECT * FROM cases", ()
        if status:
            q, args = q + " WHERE status=?", (status,)
        with self.lock:
            return [dict(r) for r in self.con.execute(q + " ORDER BY score DESC LIMIT ?", (*args, limit))]

    def feedback(self, case_id: int, label: str, analyst: str, note: str):
        with self.lock, self.con:
            cur = self.con.execute(
                "UPDATE cases SET status='closed', label=?, analyst=?, note=?, reviewed_at=? WHERE case_id=?",
                (label, analyst, note, time.time(), case_id))
            return cur.rowcount > 0

    def labelled_feedback(self) -> pd.DataFrame:       # analyst labels: input for the next retrain
        with self.lock:
            return pd.read_sql_query("SELECT txn_id, label FROM cases WHERE label IS NOT NULL", self.con)

    def stats(self) -> dict:
        with self.lock:
            q = lambda s: self.con.execute(s).fetchone()[0] or 0
            lat = pd.read_sql_query("SELECT latency_ms, e2e_ms FROM scored ORDER BY rowid DESC LIMIT 5000", self.con)
            n, alerts = q("SELECT COUNT(*) FROM scored"), q("SELECT COUNT(*) FROM scored WHERE is_alert=1")
            tp = q("SELECT COUNT(*) FROM scored WHERE is_alert=1 AND sim_label=1")
            fn = q("SELECT COUNT(*) FROM scored WHERE is_alert=0 AND sim_label=1")
            labelled_alerts = q("SELECT COUNT(*) FROM scored WHERE is_alert=1 AND sim_label IS NOT NULL")
            out = {
                "scored": n, "alerts": alerts, "open_cases": q("SELECT COUNT(*) FROM cases WHERE status='open'"),
                "closed_cases": q("SELECT COUNT(*) FROM cases WHERE status='closed'"),
                "confirmed_fraud": q("SELECT COUNT(*) FROM cases WHERE label='fraud'"),
                "sim_alert_precision": (tp / labelled_alerts) if labelled_alerts else None,
                "sim_recall": (tp / (tp + fn)) if tp + fn else None,
            }
        for col in ("latency_ms", "e2e_ms"):
            s = lat[col].dropna()
            out[col] = {f"p{p}": float(s.quantile(p / 100)) for p in (50, 95, 99)} if len(s) else None
        return out
