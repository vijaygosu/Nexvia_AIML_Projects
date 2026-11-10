import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pydicom")

from medimg.data.dicom_utils import array_to_dicom, dicom_to_bytes, dicom_to_uint8, read_dicom, summarize


def test_roundtrip_and_metadata_hashing():
    img = (np.random.rand(64, 48) * 255).astype(np.uint8)
    ds = read_dicom(dicom_to_bytes(array_to_dicom(img, patient_id="P123", patient_name="Doe^Jane")))
    out = dicom_to_uint8(ds)
    assert out.shape == (64, 48)
    meta = summarize(ds)
    assert meta["modality"] == "CR" and meta["rows"] == 64
    assert "P123" not in str(meta) and "Doe" not in str(meta)  # no direct identifiers kept


def test_monochrome1_is_inverted():
    img = np.tile(np.arange(0, 256, 4, dtype=np.uint8), (64, 1))
    ds = array_to_dicom(img)
    a = dicom_to_uint8(read_dicom(dicom_to_bytes(ds)))
    ds.PhotometricInterpretation = "MONOCHROME1"
    b = dicom_to_uint8(read_dicom(dicom_to_bytes(ds)))
    assert a[0, 0] < a[0, -1] and b[0, 0] > b[0, -1]
