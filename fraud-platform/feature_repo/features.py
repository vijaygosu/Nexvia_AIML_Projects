from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Array, Float64, Int64

user = Entity(name="user_id", join_keys=["user_id"])

batch = FileSource(path="data/user_state.parquet", timestamp_field="event_timestamp",
                   created_timestamp_column="created")
push = PushSource(name="user_state_push", batch_source=batch)

# Per-account running aggregates. fraud.features.UserState turns these into the model's
# features at request time, with the SAME code used to build the training set.
user_state = FeatureView(
    name="user_state", entities=[user], ttl=timedelta(days=365), source=push,
    schema=[Field(name="n", dtype=Int64), Field(name="total", dtype=Float64),
            Field(name="last_ts", dtype=Int64), Field(name="last_device", dtype=Int64),
            Field(name="last_location", dtype=Int64), Field(name="recent_ts", dtype=Array(Int64))])
