"""DICOM helpers: read pixels with correct windowing/photometric handling, wrap images as DICOM."""
import hashlib
import io

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CR_STORAGE = "1.2.840.10008.5.1.4.1.1.1"  # Computed Radiography Image Storage


class UnsupportedStudy(Exception):
    pass


def _f(x):
    try:
        return float(x)
    except TypeError:
        return float(x[0])


def read_dicom(data: bytes):
    return pydicom.dcmread(io.BytesIO(data), force=True)


def dicom_to_uint8(ds) -> np.ndarray:
    """Return a 2D uint8 image with rescale, windowing and MONOCHROME1 inversion applied."""
    if not hasattr(ds, "PixelData"):
        raise UnsupportedStudy("no pixel data")
    if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
        raise UnsupportedStudy("multi-frame images are not supported by the 2D chest X-ray pipeline")
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3].mean(-1)
        else:
            raise UnsupportedStudy("3D volume: use the segmentation pipeline")
    arr = arr * _f(getattr(ds, "RescaleSlope", 1)) + _f(getattr(ds, "RescaleIntercept", 0))
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        c, w = _f(ds.WindowCenter), _f(ds.WindowWidth)
        lo, hi = c - w / 2, c + w / 2
    else:
        lo, hi = np.percentile(arr, (0.5, 99.5))
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = 1 - arr
    return (arr * 255).astype(np.uint8)


def summarize(ds) -> dict:
    """Non-identifying metadata only. Patient name is never stored; the patient ID is hashed."""
    pid = str(getattr(ds, "PatientID", ""))
    spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    return {
        "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "patient_ref": hashlib.sha256(pid.encode()).hexdigest()[:10] if pid else "",
        "modality": str(getattr(ds, "Modality", "")),
        "view_position": str(getattr(ds, "ViewPosition", "")),
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
        "pixel_spacing": [float(v) for v in spacing] if spacing else None,
        "study_description": str(getattr(ds, "StudyDescription", "")),
    }


def array_to_dicom(img: np.ndarray, patient_id="DEMO-0001", patient_name="Demo^Patient",
                   study_description="Chest X-ray (demo)", view="PA") -> Dataset:
    """Wrap a 2D uint8 array as a valid CR DICOM object (for demos/tests with PNG datasets)."""
    assert img.ndim == 2 and img.dtype == np.uint8
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CR_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = CR_STORAGE, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.PatientName, ds.PatientID = patient_name, patient_id
    ds.Modality, ds.StudyDescription, ds.ViewPosition = "CR", study_description, view
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.Rows, ds.Columns = img.shape
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.PixelData = img.tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    return ds


def dicom_to_bytes(ds: Dataset) -> bytes:
    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()
