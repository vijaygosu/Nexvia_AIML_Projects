from medimg.data.splits import assign_split, keep_patient


def test_deterministic():
    assert assign_split("123") == assign_split("123")
    assert assign_split("123", seed=1) in {"train", "val", "test"}


def test_proportions_roughly_match():
    ids = [str(i) for i in range(20000)]
    counts = {"train": 0, "val": 0, "test": 0}
    for i in ids:
        counts[assign_split(i)] += 1
    assert 0.77 < counts["train"] / len(ids) < 0.83
    assert 0.07 < counts["val"] / len(ids) < 0.13
    assert 0.07 < counts["test"] / len(ids) < 0.13


def test_patient_never_in_two_splits():
    # every image of a patient carries the same patient id -> same split by construction
    for pid in range(500):
        assert len({assign_split(str(pid)) for _ in range(5)}) == 1


def test_subsample_fraction():
    kept = sum(keep_patient(str(i), 0.3) for i in range(20000)) / 20000
    assert 0.27 < kept < 0.33
