"""FastAPI scoring service.

POST /score                 score one transaction (sync path)
GET  /alerts                analyst queue            POST /alerts/{id}/feedback  analyst label
GET  /stats  /drift  /health
Kafka consumer thread (if KAFKA_BOOTSTRAP is set): topic 'transactions' -> score -> topic 'fraud-scores'.
"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..serving import Scorer
from .db import DB

TOPIC_IN, TOPIC_OUT = os.getenv("TOPIC_IN", "transactions"), os.getenv("TOPIC_OUT", "fraud-scores")


class Txn(BaseModel):
    txn_id: int
    ts: int = Field(..., description="seconds since epoch/reference")
    user_id: int
    amount: float = Field(..., gt=0)
    device_id: int = -1
    email_domain: int = -1
    location: int = -1
    product: int = 0


class Feedback(BaseModel):
    label: str = Field(..., pattern="^(fraud|legit)$")
    analyst: str = "analyst"
    note: str = ""


def handle(app, txn: dict, sim_label=None, produced_at=None):
    res = app.state.scorer.score(txn)
    e2e = (time.time() - produced_at) * 1000 if produced_at else None
    app.state.db.log(res, sim_label, e2e)
    return res


def consume_loop(app, stop: threading.Event):
    from kafka import KafkaConsumer, KafkaProducer
    servers = os.environ["KAFKA_BOOTSTRAP"]
    while not stop.is_set():                              # broker may still be starting
        try:
            cons = KafkaConsumer(TOPIC_IN, bootstrap_servers=servers, group_id="fraud-scorer",
                                 auto_offset_reset="earliest", value_deserializer=lambda b: json.loads(b))
            prod = KafkaProducer(bootstrap_servers=servers, value_serializer=lambda v: json.dumps(v).encode())
            break
        except Exception as e:
            print("kafka not ready:", e)
            time.sleep(3)
    else:
        return
    while not stop.is_set():
        for _, msgs in cons.poll(timeout_ms=500).items():
            for m in msgs:
                v = m.value
                res = handle(app, {k: v[k] for k in ("txn_id", "ts", "user_id", "amount", "device_id",
                                                     "email_domain", "location", "product")},
                             v.get("is_fraud"), v.get("produced_at"))
                prod.send(TOPIC_OUT, {k: res[k] for k in ("txn_id", "user_id", "fraud_score", "is_alert", "latency_ms")})


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = Scorer.from_dir(os.getenv("MODEL_DIR", "models"))
    app.state.db = DB(os.getenv("DB_PATH", "data/cases.db"))
    stop = threading.Event()
    if os.getenv("KAFKA_BOOTSTRAP"):
        threading.Thread(target=consume_loop, args=(app, stop), daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="Fraud Scoring API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": app.state.scorer.threshold, "blend_w": app.state.scorer.w}


@app.post("/score")
def score(t: Txn):
    r = handle(app, t.model_dump())
    return {k: r[k] for k in ("txn_id", "fraud_score", "xgb_score", "gnn_score", "is_alert", "latency_ms")}


@app.get("/alerts")
def alerts(status: str | None = Query(None, pattern="^(open|closed)$"), limit: int = 200):
    return app.state.db.alerts(status, limit)


@app.post("/alerts/{case_id}/feedback")
def feedback(case_id: int, fb: Feedback):
    if not app.state.db.feedback(case_id, fb.label, fb.analyst, fb.note):
        raise HTTPException(404, "case not found")
    return {"case_id": case_id, "status": "closed", "label": fb.label}


@app.get("/stats")
def stats():
    return app.state.db.stats()


@app.get("/drift")
def drift():
    p = os.path.join(os.getenv("REPORT_DIR", "monitoring/reports"), "latest.json")
    return json.load(open(p)) if os.path.exists(p) else {"status": "no drift report yet"}


@app.post("/admin/reload")
def reload_model():
    """Hot-load freshly retrained artifacts from MODEL_DIR (called by the drift job after retraining)."""
    app.state.scorer = Scorer.from_dir(os.getenv("MODEL_DIR", "models"))
    return {"status": "reloaded", "threshold": app.state.scorer.threshold, "blend_w": app.state.scorer.w}
