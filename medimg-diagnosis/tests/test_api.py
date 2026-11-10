import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from medimg.data.dicom_utils import array_to_dicom, dicom_to_bytes
from medimg.service.app import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(data_dir=str(tmp_path_factory.mktemp("svc")), ckpt=None, orthanc_url=None)
    with TestClient(app) as c:
        yield c


def _dicom(modality="CR"):
    ds = array_to_dicom((np.random.rand(128, 128) * 255).astype(np.uint8))
    ds.Modality = modality
    return dicom_to_bytes(ds)


def test_end_to_end_review_flow(client):
    h = client.get("/health").json()
    assert h["model_trained"] is False and "NOT FOR CLINICAL USE" in h["disclaimer"]

    r = client.post("/predict", files={"file": ("a.dcm", _dicom())})
    assert r.status_code == 200
    sid = r.json()["id"]

    s = client.get(f"/studies/{sid}").json()
    assert s["status"] == "pending" and s["top_label"] in s["predictions"] and "disclaimer" in s
    assert client.get(f"/studies/{sid}/image.png").headers["content-type"] == "image/png"
    assert client.get(f"/studies/{sid}/heatmap.png").status_code == 200
    assert client.get(f"/studies/{sid}/heatmap.png", params={"label": "Mass"}).status_code == 200
    assert client.get(f"/studies/{sid}/heatmap.png", params={"label": "Nope"}).status_code == 400

    body = {"decision": "approve", "comment": "agree"}
    assert client.post(f"/studies/{sid}/decision", json=body).status_code == 401  # no reviewer identity
    assert client.post(f"/studies/{sid}/decision", json=body, headers={"X-User": "dr.lee"}).status_code == 200
    assert client.post(f"/studies/{sid}/decision", json=body, headers={"X-User": "dr.lee"}).status_code == 409

    actions = [e["action"] for e in client.get("/audit", params={"study_id": sid}).json()["data"]]
    assert {"AI_RESULT", "VIEW", "DECISION"} <= set(actions)
    assert client.get("/audit/verify").json()["ok"] is True
    assert client.get("/studies/x").status_code == 404


def test_unsupported_modality_is_rejected_and_audited(client):
    r = client.post("/predict", files={"file": ("ct.dcm", _dicom("CT"))})
    assert r.status_code == 415
    assert any(e["action"] == "SKIPPED" for e in client.get("/audit").json()["data"])


def test_duplicate_upload_is_idempotent(client):
    data = _dicom()
    a = client.post("/predict", files={"file": ("a.dcm", data)}).json()
    b = client.post("/predict", files={"file": ("a.dcm", data)}).json()
    assert a["id"] == b["id"] and b["duplicate"] is True
