"""Append-only, hash-chained audit log in SQLite.

* UPDATE/DELETE are blocked by triggers (defence in depth, not a security boundary: whoever owns the
  DB file can drop them - hence the hash chain, which makes tampering detectable by verify()).
* Each row's hash = SHA-256(prev_hash | ts | user | action | study_id | details_json).
* Truncating the *tail* of the log is only detectable if the head hash was anchored elsewhere
  (export GET /audit/head periodically).
"""
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, user TEXT NOT NULL, action TEXT NOT NULL, study_id TEXT,
  details TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""


def compute_hash(prev, ts, user, action, study_id, details_json) -> str:
    payload = "|".join([prev, ts, user, action, study_id or "", details_json])
    return hashlib.sha256(payload.encode()).hexdigest()


class AuditLog:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with closing(self._conn()) as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=30, isolation_level=None)

    def append(self, user: str, action: str, study_id=None, details=None) -> dict:
        details_json = json.dumps(details or {}, sort_keys=True, separators=(",", ":"))
        ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with closing(self._conn()) as c:
            c.execute("BEGIN IMMEDIATE")  # serialises writers so the chain cannot fork
            try:
                row = c.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
                prev = row[0] if row else GENESIS
                h = compute_hash(prev, ts, user, action, study_id, details_json)
                cur = c.execute(
                    "INSERT INTO audit (ts,user,action,study_id,details,prev_hash,hash) VALUES (?,?,?,?,?,?,?)",
                    (ts, user, action, study_id, details_json, prev, h))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        return {"seq": cur.lastrowid, "hash": h}

    def entries(self, limit=200, study_id=None):
        q, args = "SELECT seq,ts,user,action,study_id,details,prev_hash,hash FROM audit", []
        if study_id:
            q += " WHERE study_id=?"
            args.append(study_id)
        q += " ORDER BY seq DESC LIMIT ?"
        args.append(limit)
        with closing(self._conn()) as c:
            cols = ["seq", "ts", "user", "action", "study_id", "details", "prev_hash", "hash"]
            return [dict(zip(cols, r)) | {"details": json.loads(r[5])} for r in c.execute(q, args)]

    def head(self) -> str:
        with closing(self._conn()) as c:
            row = c.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS

    def verify(self) -> dict:
        prev, n = GENESIS, 0
        with closing(self._conn()) as c:
            for seq, ts, user, action, sid, details, p, h in c.execute(
                    "SELECT seq,ts,user,action,study_id,details,prev_hash,hash FROM audit ORDER BY seq"):
                if p != prev or compute_hash(p, ts, user, action, sid, details) != h:
                    return {"ok": False, "first_bad_seq": seq, "checked": n}
                prev, n = h, n + 1
        return {"ok": True, "first_bad_seq": None, "checked": n, "head": prev}
