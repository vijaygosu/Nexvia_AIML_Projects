"""FastAPI inference + review service.  Run:  uvicorn --factory medimg.service.app:create_app"""
import hashlib
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from medimg.config import DISCLAIMER
from medimg.data.dicom_utils import UnsupportedStudy, dicom_to_uint8, read_dicom, summarize
from medimg.service.audit import AuditLog
from medimg.service.db import Store
from medimg.service.orthanc import OrthancClient, Poller
from medimg.service.pipeline import InferenceEngine

log = logging.getLogger("medimg.service")
ALLOWED = {m.strip() for m in os.getenv("ALLOWED_MODALITIES", "CR,DX").split(",")}


class DecisionIn(BaseModel):
    decision: Literal["approve", "reject"]
    comment: str = Field("", max_length=2000)


def _wrap(payload):
    return {"disclaimer": DISCLAIMER, **payload} if isinstance(payload, dict) else {"disclaimer": DISCLAIMER, "data": payload}


def create_app(data_dir=None, ckpt=None, orthanc_url=None, start_poller=True) -> FastAPI:
    data_dir = data_dir or os.getenv("DATA_DIR", "./service_data")
    ckpt = ckpt if ckpt is not None else os.getenv("MODEL_CKPT")
    orthanc_url = orthanc_url if orthanc_url is not None else os.getenv("ORTHANC_URL")
    img_dir = os.path.join(data_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    store = Store(os.path.join(data_dir, "app.db"))
    audit = AuditLog(os.path.join(data_dir, "audit.db"))
    engine = InferenceEngine(ckpt)
    lock = threading.Lock()  # one inference at a time (single model instance)

    def ingest(data: bytes, source: str) -> dict:
        ds = read_dicom(data)
        meta = summarize(ds)
        sid = hashlib.sha1(meta["sop_uid"].encode()).hexdigest()[:16]
        if store.get(sid):
            return {"id": sid, "duplicate": True}
        if meta["modality"] not in ALLOWED:
            audit.append("system", "SKIPPED", sid, {"reason": f"modality {meta['modality']} not supported", "source": source})
            raise HTTPException(415, f"modality {meta['modality']!r} not supported (allowed: {sorted(ALLOWED)})")
        try:
            gray = dicom_to_uint8(ds)
        except UnsupportedStudy as e:
            audit.append("system", "SKIPPED", sid, {"reason": str(e), "source": source})
            raise HTTPException(415, str(e))
        with lock:
            res = engine.predict(gray)
        for name, key in (("image", "image_png"), ("heatmap", "heatmap_png")):
            with open(os.path.join(img_dir, f"{sid}_{name}.png"), "wb") as f:
                f.write(res[key])
        store.insert({
            "id": sid, "sop_uid": meta["sop_uid"], "study_uid": meta["study_uid"], "patient_ref": meta["patient_ref"],
            "modality": meta["modality"], "received_at": datetime.now(timezone.utc).isoformat(),
            "model_version": res["model_version"], "predictions": json.dumps(res["predictions"]),
            "top_label": res["top_label"], "top_prob": res["top_prob"], "meta": json.dumps(meta)})
        audit.append("system", "AI_RESULT", sid, {
            "model_version": res["model_version"], "top_label": res["top_label"], "top_prob": res["top_prob"],
            "source": source, "predictions": res["predictions"]})
        return {"id": sid, "duplicate": False}

    def on_orthanc(instance_id, data):
        try:
            ingest(data, f"orthanc:{instance_id}")
        except HTTPException:
            pass  # already audited as SKIPPED

    @asynccontextmanager
    async def lifespan(app):
        poller = None
        if orthanc_url and start_poller:
            poller = Poller(OrthancClient(orthanc_url, _auth()), store, on_orthanc,
                            float(os.getenv("POLL_INTERVAL", "3")))
            poller.start()
        yield
        if poller:
            poller.stop()

    def _auth():
        u, p = os.getenv("ORTHANC_USER"), os.getenv("ORTHANC_PASSWORD")
        return (u, p) if u else None

    app = FastAPI(title="MedImg AI Review Service", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def disclaimer_header(request, call_next):
        resp = await call_next(request)
        resp.headers["X-Disclaimer"] = "For education only - not for clinical use"
        return resp

    def _study(sid):
        s = store.get(sid)
        if not s:
            raise HTTPException(404, "study not found")
        return s

    @app.get("/health")
    def health():
        return _wrap({"status": "ok", "model_version": engine.version, "model_trained": engine.trained,
                      "labels": engine.labels, "orthanc": bool(orthanc_url)})

    @app.post("/predict")
    def predict(file: UploadFile = File(...), x_user: str = Header(default="")):
        out = ingest(file.file.read(), f"upload:{x_user or 'anonymous'}")
        return _wrap(out)

    @app.get("/studies")
    def studies(status: str | None = None):
        keys = ("id", "received_at", "patient_ref", "modality", "top_label", "top_prob", "status", "model_version")
        return _wrap([{k: s[k] for k in keys} for s in store.list(status)])

    @app.get("/studies/{sid}")
    def study(sid: str, x_user: str = Header(default="")):
        s = _study(sid)
        audit.append(x_user or "anonymous", "VIEW", sid, {})
        return _wrap(s)

    @app.get("/studies/{sid}/image.png")
    def image(sid: str):
        _study(sid)
        return FileResponse(os.path.join(img_dir, f"{sid}_image.png"), media_type="image/png")

    @app.get("/studies/{sid}/heatmap.png")
    def heatmap(sid: str, label: str | None = None):
        s = _study(sid)
        if label and label != s["top_label"]:
            if label not in engine.labels:
                raise HTTPException(400, f"unknown label {label!r}")
            from PIL import Image
            import numpy as np
            gray = np.asarray(Image.open(os.path.join(img_dir, f"{sid}_image.png")).convert("L"))
            with lock:
                return Response(engine.heatmap_png(gray, label), media_type="image/png")
        return FileResponse(os.path.join(img_dir, f"{sid}_heatmap.png"), media_type="image/png")

    @app.post("/studies/{sid}/decision")
    def decision(sid: str, body: DecisionIn, x_user: str = Header(default="")):
        if not x_user.strip():
            raise HTTPException(401, "X-User header (reviewer identity) is required")
        s = _study(sid)
        status = "approved" if body.decision == "approve" else "rejected"
        ts = datetime.now(timezone.utc).isoformat()
        if not store.decide(sid, status, x_user.strip(), ts, body.comment):
            raise HTTPException(409, f"study already {s['status']}")
        audit.append(x_user.strip(), "DECISION", sid, {
            "decision": body.decision, "comment": body.comment, "model_version": s["model_version"],
            "top_label": s["top_label"], "top_prob": s["top_prob"], "predictions": s["predictions"]})
        return _wrap({"id": sid, "status": status})

    @app.get("/audit")
    def audit_list(limit: int = 200, study_id: str | None = None):
        return _wrap(audit.entries(min(limit, 1000), study_id))

    @app.get("/audit/verify")
    def audit_verify():
        return _wrap(audit.verify())

    @app.get("/audit/head")
    def audit_head():
        return _wrap({"head": audit.head()})

    return app
