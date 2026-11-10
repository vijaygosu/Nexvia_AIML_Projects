# Suggested ownership split (edit to match your team)
| Area | Files |
|---|---|
| Data & DVC | `src/medimg/data/*`, `dvc.yaml`, `params.yaml` |
| Classification & explainability | `models/classifier.py`, `explain/gradcam.py`, `train/train_classifier.py`, `train/evaluate.py` |
| Segmentation (MONAI) | `models/segmentation.py`, `train/train_segmentation.py` |
| DICOM, Orthanc & service | `data/dicom_utils.py`, `service/*`, `orthanc/`, `scripts/` |
| Viewer, audit & safety UX | `viewer/*`, `service/audit.py` |
| Docker, CI, docs & model card | `Dockerfile.service`, `docker-compose.yml`, `.github/`, `docs/*` |
