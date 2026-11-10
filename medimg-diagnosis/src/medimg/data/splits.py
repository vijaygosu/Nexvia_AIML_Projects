"""Deterministic, patient-level splitting (no patient ever appears in two splits)."""
import hashlib


def _bucket(key: str, seed: int, mod: int) -> int:
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16) % mod


def assign_split(patient_id: str, val_pct: int = 10, test_pct: int = 10, seed: int = 0) -> str:
    h = _bucket(str(patient_id), seed, 100)
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def keep_patient(patient_id: str, fraction: float, seed: int = 0) -> bool:
    """Subsample whole patients (never single images) to build a smaller dataset."""
    return _bucket(str(patient_id), seed + 1, 10000) < int(fraction * 10000)
