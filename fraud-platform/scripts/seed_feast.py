"""Convert models/user_state.csv -> parquet, `feast apply`, `feast materialize`."""
import json
import subprocess
import sys

import pandas as pd

s = pd.read_csv("models/user_state.csv")
s["recent_ts"] = s["recent_ts"].apply(json.loads)
now = pd.Timestamp.utcnow().tz_localize(None)
s["event_timestamp"], s["created"] = now, now
import os; os.makedirs("feature_repo/data", exist_ok=True)
s.to_parquet("feature_repo/data/user_state.parquet", index=False)
for cmd in (["feast", "apply"], ["feast", "materialize-incremental", (now + pd.Timedelta(minutes=1)).isoformat()]):
    subprocess.run(cmd, cwd="feature_repo", check=True)
print("Feast online store seeded with", len(s), "accounts. Run the API with STATE_BACKEND=feast")
