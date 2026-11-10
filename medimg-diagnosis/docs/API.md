# API (service :8000, proxied at `/api` by the viewer)

All JSON responses include a `disclaimer` field; all responses carry `X-Disclaimer`.
Decisions require an `X-User` header (reviewer identity; demo-grade auth).

| Method & path | Description |
|---|---|
| `GET /health` | model version, `model_trained`, labels |
| `POST /predict` (multipart `file`=DICOM) | ingest without Orthanc. 415 for non-CR/DX or multi-frame/3D |
| `GET /studies?status=pending\|approved\|rejected` | list |
| `GET /studies/{id}` | detail incl. all label scores. Writes a `VIEW` audit entry |
| `GET /studies/{id}/image.png` | web-resolution image |
| `GET /studies/{id}/heatmap.png[?label=Mass]` | RGBA Grad-CAM (top class stored; others computed on demand) |
| `POST /studies/{id}/decision` `{decision:"approve"\|"reject", comment}` | 401 without `X-User`, 409 if already decided |
| `GET /audit?limit=&study_id=` | newest first |
| `GET /audit/verify` | recompute hash chain `{ok, first_bad_seq, checked}` |
| `GET /audit/head` | latest hash (anchor this externally) |

Interactive docs: `http://localhost:8000/docs`.
