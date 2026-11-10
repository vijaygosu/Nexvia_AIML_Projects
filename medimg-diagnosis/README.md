# AI-Assisted Medical Imaging & Diagnosis System

> **FOR EDUCATION ONLY - NOT FOR CLINICAL USE.** Research prototype, not clinically validated.

Chest X-rays are stored in an **Orthanc** DICOM server, picked up by an inference service
(**PyTorch** DenseNet121 + **Grad-CAM**), shown in a **React** viewer with a heatmap overlay and confidence
score, and reviewed by a radiologist who **Approves/Rejects** the AI finding. Every action lands in a
tamper-evident, hash-chained **audit log**. A **MONAI** 3D U-Net (Medical Segmentation Decathlon) covers
the segmentation requirement. Data and weights are versioned with **DVC**; everything runs with **Docker Compose**.

```
 PNG/DICOM ──> Orthanc ──(/changes poll)──> Inference service ──> SQLite (studies + audit chain)
                                              DenseNet121 + Grad-CAM        │
                                                                            v
                                                         React viewer (overlay, confidence, Approve/Reject)
```
Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/API.md](docs/API.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md)

## Quick start (demo stack, no training needed)
```bash
docker compose up --build -d          # Orthanc :8042, service :8000, viewer :8080
pip install pydicom pillow numpy requests
python scripts/upload_to_orthanc.py --src data/demo_images --orthanc http://localhost:8042   # any PNG/JPG chest X-rays
open http://localhost:8080
```
Without a checkpoint the service runs an **UNTRAINED-DEMO** model (random head) and the UI says so loudly.
Put trained weights at `models/densenet121_nih.pt` (see below) and restart `service`.

## Reproduce the experiments
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# 1. Data: download NIH ChestX-ray14 (images + Data_Entry_2017.csv) into data/raw/nih/   (needs internet; ~42 GB full)
dvc init            # first time only
dvc repro           # prepare -> train_classifier -> evaluate (-> train_segmentation)
dvc metrics show    # AUROC, Dice
```
Stages and parameters live in `dvc.yaml` / `params.yaml`. Default: ChestX-ray8 label subset, 30% of patients,
256px cache, DenseNet121 (ImageNet init), 8 epochs. A single T4 (Colab) is enough.
Add a remote with `dvc remote add -d storage <s3://... or ./local_remote>` to share data and weights.

Segmentation only: `python -m medimg.train.train_segmentation --task Task09_Spleen` (downloads ~1.5 GB).

## Tests
```bash
pip install -r requirements.txt && pytest -q
```
Covers patient-level splitting, DICOM parsing/anonymisation, Grad-CAM shapes/ranges, the audit hash chain
(tamper detection, append-only triggers, concurrency) and the full API review flow.

## Safety & ethics
Education only. Patient names are never stored (patient IDs are hashed), the UI shows a persistent banner and
watermark, API responses carry a `disclaimer` field and `X-Disclaimer` header. Known limits and bias analysis:
[docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Repo layout
`src/medimg/{data,models,explain,train,service}` · `viewer/` React app · `orthanc/` config · `docs/` ·
`tests/` · `scripts/` · `dvc.yaml`, `docker-compose.yml`

## Known gaps / next steps
* Segmentation model is trained and evaluated separately; it is **not** wired into the viewer (needs a 3D series viewer, e.g. Cornerstone3D/OHIF, and CT in Orthanc).
* Viewer is a lightweight 2D viewer (no window/level, no DICOM-SR/Secondary-Capture write-back of results).
* Auth is a reviewer-name header for demo purposes; production would need OIDC and role-based access.
* No external validation; labels in NIH are NLP-mined and noisy.
