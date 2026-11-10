"""Replays held-out transactions into Kafka (optionally injecting drift for the demo)."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/replay.csv")
    ap.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    ap.add_argument("--topic", default=os.getenv("TOPIC_IN", "transactions"))
    ap.add_argument("--rate", type=float, default=50, help="messages per second (0 = as fast as possible)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drift-after", type=int, default=0, help="after N msgs, shift amounts/devices/locations")
    a = ap.parse_args(argv)
    from kafka import KafkaProducer
    prod = KafkaProducer(bootstrap_servers=a.bootstrap, value_serializer=lambda v: json.dumps(v).encode(), linger_ms=5)
    df = pd.read_csv(a.file)
    if a.limit:
        df = df.head(a.limit)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i, r in enumerate(df.to_dict("records")):
        if a.drift_after and i >= a.drift_after:               # simulated concept/data drift
            r["amount"] = round(r["amount"] * 4, 2)
            r["device_id"] = int(900_000 + rng.integers(0, 10**6))
            r["location"] = int(rng.integers(0, 300))
        r["produced_at"] = time.time()
        prod.send(a.topic, r)
        if a.rate and i % max(int(a.rate // 10), 1) == 0:
            time.sleep(max(0.0, (i + 1) / a.rate - (time.time() - t0)))
    prod.flush()
    print(f"sent {len(df):,} messages in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
