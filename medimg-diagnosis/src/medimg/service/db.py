import json
import sqlite3
from contextlib import closing

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
  id TEXT PRIMARY KEY, sop_uid TEXT UNIQUE, study_uid TEXT, patient_ref TEXT, modality TEXT,
  received_at TEXT, model_version TEXT, predictions TEXT, top_label TEXT, top_prob REAL,
  status TEXT NOT NULL DEFAULT 'pending', decided_by TEXT, decided_at TEXT, comment TEXT, meta TEXT
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path):
        self.path = path
        with closing(self._c()) as c:
            c.executescript(SCHEMA)

    def _c(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _row(r):
        d = dict(r)
        d["predictions"] = json.loads(d["predictions"] or "{}")
        d["meta"] = json.loads(d["meta"] or "{}")
        return d

    def get(self, sid):
        with closing(self._c()) as c:
            r = c.execute("SELECT * FROM studies WHERE id=?", (sid,)).fetchone()
        return self._row(r) if r else None

    def list(self, status=None):
        q, a = "SELECT * FROM studies", []
        if status:
            q += " WHERE status=?"
            a.append(status)
        with closing(self._c()) as c:
            return [self._row(r) for r in c.execute(q + " ORDER BY received_at DESC", a)]

    def insert(self, rec: dict) -> bool:
        with closing(self._c()) as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO studies (id,sop_uid,study_uid,patient_ref,modality,received_at,model_version,"
                "predictions,top_label,top_prob,meta) VALUES (:id,:sop_uid,:study_uid,:patient_ref,:modality,"
                ":received_at,:model_version,:predictions,:top_label,:top_prob,:meta)", rec)
            c.commit()
            return cur.rowcount == 1

    def decide(self, sid, status, user, ts, comment) -> bool:
        with closing(self._c()) as c:
            cur = c.execute(
                "UPDATE studies SET status=?,decided_by=?,decided_at=?,comment=? WHERE id=? AND status='pending'",
                (status, user, ts, comment, sid))
            c.commit()
            return cur.rowcount == 1

    def get_kv(self, k, default=None):
        with closing(self._c()) as c:
            r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return r[0] if r else default

    def set_kv(self, k, v):
        with closing(self._c()) as c:
            c.execute("INSERT INTO kv (k,v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))
            c.commit()
