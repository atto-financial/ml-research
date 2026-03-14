from datetime import timedelta
import pandas as pd
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# Define an entity for the user
user = Entity(name="user_id", join_keys=["user_id"])

# Define the source (using the dbt model output if possible, or a parquet for now)
# In production, this would be a PostgreSQLSource pointing to the dbt model table
loan_features_source = FileSource(
    name="loan_features_source",
    path="data/loan_features.parquet", # Path to the offline data
    timestamp_field="event_timestamp",
)

# Define the Feature View
loan_features_fv = FeatureView(
    name="loan_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="fht1", dtype=Int64),
        Field(name="fht2", dtype=Int64),
        Field(name="fht3", dtype=Int64),
        Field(name="fht4", dtype=Int64),
        Field(name="fht5", dtype=Int64),
        Field(name="fht6", dtype=Int64),
        Field(name="fht7", dtype=Int64),
        Field(name="fht8", dtype=Int64),
        Field(name="kmsi1", dtype=Int64),
        Field(name="kmsi2", dtype=Int64),
        Field(name="kmsi3", dtype=Int64),
        Field(name="kmsi4", dtype=Int64),
        Field(name="kmsi5", dtype=Int64),
        Field(name="kmsi6", dtype=Int64),
        Field(name="kmsi7", dtype=Int64),
        Field(name="kmsi8", dtype=Int64),
        Field(name="ust", dtype=Int64),
    ],
    online=True,
    source=loan_features_source,
    tags={"team": "loan_approval"},
)
