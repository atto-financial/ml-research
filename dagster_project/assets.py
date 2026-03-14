from dagster import asset, AssetExecutionContext
from app.data.data_loading import data_loading_fsk_v1
import pandas as pd

@asset(group_name="data_pipeline")
def fsk_engineered_data(context: AssetExecutionContext) -> pd.DataFrame:
    """
    Load fully engineered data directly from the dbt model (fct_loan_features).
    All cleaning, mapping, and engineering are now handled in the SQL/dbt layer.
    """
    data = data_loading_fsk_v1()
    if data is not None:
        context.log.info(f"Loaded {len(data)} rows of engineered features")
        context.log.info(f"Columns: {data.columns.tolist()}")
    else:
        context.log.error("Failed to load engineered data")
    return data
