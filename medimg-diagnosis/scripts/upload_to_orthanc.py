"""Wrap PNG/JPG chest X-rays as DICOM and push them to Orthanc (demo helper).

  python scripts/upload_to_orthanc.py --src data/demo_images --orthanc http://localhost:8042
Images from NIH/CheXpert are PNG/JPG, so this also demonstrates DICOM structure (patient/study/series UIDs,
photometric interpretation, pixel data). Patient IDs are synthetic.
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from medimg.data.dicom_utils import array_to_dicom, dicom_to_bytes  # noqa: E402
from medimg.service.orthanc import OrthancClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--orthanc", default="http://localhost:8042")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--save-dcm-dir", default=None)
    a = ap.parse_args()
    files = sorted(sum((glob.glob(os.path.join(a.src, e)) for e in ("*.png", "*.jpg", "*.jpeg")), []))[: a.limit]
    client = OrthancClient(a.orthanc)
    for i, f in enumerate(files):
        arr = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
        ds = array_to_dicom(arr, patient_id=f"DEMO-{i:04d}", patient_name=f"Demo^Patient{i:04d}")
        data = dicom_to_bytes(ds)
        if a.save_dcm_dir:
            os.makedirs(a.save_dcm_dir, exist_ok=True)
            open(os.path.join(a.save_dcm_dir, f"demo_{i:04d}.dcm"), "wb").write(data)
        print(f, "->", client.upload(data).get("ID"))


if __name__ == "__main__":
    main()
