"""Minimal Orthanc REST client and a change-feed poller."""
import logging
import threading
import time

import requests

log = logging.getLogger("medimg.orthanc")


class OrthancClient:
    def __init__(self, base_url: str, auth=None, timeout=30):
        self.base, self.auth, self.timeout = base_url.rstrip("/"), auth, timeout

    def changes(self, since: int = 0, limit: int = 50) -> dict:
        r = requests.get(f"{self.base}/changes", params={"since": since, "limit": limit},
                         auth=self.auth, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def instance_file(self, instance_id: str) -> bytes:
        r = requests.get(f"{self.base}/instances/{instance_id}/file", auth=self.auth, timeout=self.timeout)
        r.raise_for_status()
        return r.content

    def upload(self, dicom_bytes: bytes) -> dict:
        r = requests.post(f"{self.base}/instances", data=dicom_bytes, auth=self.auth, timeout=self.timeout,
                          headers={"Content-Type": "application/dicom"})
        r.raise_for_status()
        return r.json()


class Poller(threading.Thread):
    """Follows Orthanc's /changes feed and calls handler(instance_id, dicom_bytes) for each NewInstance.
    The cursor is persisted so a restart does not reprocess old studies; ingestion is idempotent anyway."""

    def __init__(self, client, store, handler, interval=3.0):
        super().__init__(daemon=True, name="orthanc-poller")
        self.client, self.store, self.handler, self.interval = client, store, handler, interval
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def poll_once(self) -> int:
        since = int(self.store.get_kv("orthanc_since", 0))
        resp = self.client.changes(since)
        n = 0
        for ch in resp.get("Changes", []):
            if ch.get("ChangeType") == "NewInstance":
                try:
                    self.handler(ch["ID"], self.client.instance_file(ch["ID"]))
                    n += 1
                except Exception:  # keep the feed moving; failures are audited by the handler
                    log.exception("failed to process instance %s", ch["ID"])
        self.store.set_kv("orthanc_since", resp.get("Last", since))
        return n

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.poll_once()
            except requests.RequestException as e:
                log.warning("Orthanc unreachable: %s", e)
            self._stop_evt.wait(self.interval)
