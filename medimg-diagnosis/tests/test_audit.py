import sqlite3
import threading

import pytest

from medimg.service.audit import GENESIS, AuditLog


@pytest.fixture
def log(tmp_path):
    return AuditLog(str(tmp_path / "audit.db"))


def test_chain_links_and_verifies(log):
    a = log.append("system", "AI_RESULT", "s1", {"p": 0.9})
    b = log.append("dr.x", "DECISION", "s1", {"decision": "approve"})
    assert a["hash"] != b["hash"]
    rows = log.entries()
    assert rows[1]["prev_hash"] == GENESIS and rows[0]["prev_hash"] == rows[1]["hash"]
    assert log.verify()["ok"] and log.verify()["checked"] == 2


def test_update_and_delete_are_blocked(log):
    log.append("u", "A", "s", {})
    c = sqlite3.connect(log.db_path)
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("UPDATE audit SET user='evil'")
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("DELETE FROM audit")


def test_tampering_is_detected_even_if_triggers_removed(log):
    for i in range(3):
        log.append("u", "A", f"s{i}", {"i": i})
    c = sqlite3.connect(log.db_path, isolation_level=None)
    c.execute("DROP TRIGGER audit_no_update")
    c.execute("UPDATE audit SET details='{\"i\":99}' WHERE seq=2")
    res = log.verify()
    assert not res["ok"] and res["first_bad_seq"] == 2


def test_deleting_a_middle_row_is_detected(log):
    for i in range(3):
        log.append("u", "A", f"s{i}", {})
    c = sqlite3.connect(log.db_path, isolation_level=None)
    c.execute("DROP TRIGGER audit_no_delete")
    c.execute("DELETE FROM audit WHERE seq=2")
    assert not log.verify()["ok"]


def test_concurrent_appends_keep_chain_intact(log):
    def work(n):
        for i in range(10):
            log.append(f"u{n}", "A", "s", {"i": i})
    ts = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    res = log.verify()
    assert res["ok"] and res["checked"] == 40
