from dagster import asset, AssetExecutionContext, MetadataValue
from app.data.data_loading import data_loading_fsk_v1
from app.models.lucis import train_model, select_top_features, ModelConfig
from app.models.Utils_statistics import compute_correlations
from app.utils.utils_mlflow import init_mlflow, log_model_to_mlflow
from app.utils.utils_model import setup_paths, ensure_paths, save_all_artifacts, features_importance
from app.utils.duckdb_utils import sync_to_duckdb, query_duckdb
from app.models import features_hamilton
from hamilton import driver
from datetime import datetime
import pandas as pd
import numpy as np
import os

@asset(group_name="data_pipeline")
def database_metadata(context: AssetExecutionContext) -> dict:
    """
    Dynamically inspect the table schema to identify features.
    """
    from app.utils.db_connection import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'fct_loan_features'
    """)
    columns = cursor.fetchall()
    conn.close()
    
    metadata = {col[0]: col[1] for col in columns}
    context.add_output_metadata({
        "total_columns": len(metadata),
        "columns": MetadataValue.json(list(metadata.keys()))
    })
    return metadata

@asset(group_name="data_pipeline")
def fsk_local_cache(context: AssetExecutionContext, database_metadata: dict) -> bool:
    """
    Sync PostgreSQL data to local DuckDB cache to save resources.
    """
    query = "SELECT * FROM fct_loan_features"
    success = sync_to_duckdb("fct_loan_features", query)
    if success:
        context.log.info("Successfully cached data to DuckDB")
    return success

@asset(group_name="data_pipeline")
def fsk_engineered_data(context: AssetExecutionContext, fsk_local_cache: bool) -> pd.DataFrame:
    """
    Load data from local DuckDB cache.
    """
    if not fsk_local_cache:
        context.log.warning("Cache failed, falling back to PostgreSQL")
        return data_loading_fsk_v1()
        
    data = query_duckdb("SELECT * FROM fct_loan_features")
    context.log.info(f"Loaded {len(data)} rows from DuckDB")
    return data

@asset(group_name="ml_pipeline")
def preprocessed_features(context: AssetExecutionContext, fsk_engineered_data: pd.DataFrame):
    """
    Use Hamilton to modularly preprocess features.
    """
    config = {
        'df': fsk_engineered_data
    }
    adapter = driver.Driver(config, features_hamilton)
    
    # Run the DAG to get preprocessed data and the scaler
    results = adapter.execute(['final_preprocessed_data', 'feature_scaler'])
    
    df_preprocessed = results['final_preprocessed_data']
    scaler = results['feature_scaler']
    
    context.log.info(f"Preprocessing complete. Columns: {len(df_preprocessed.columns)}")
    return {"df": df_preprocessed, "scaler": scaler}

@asset(group_name="ml_pipeline")
def lucis_model(context: AssetExecutionContext, preprocessed_features: dict):
    """
    Train Lucis model (Random Forest) using preprocessed data.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = ModelConfig()
    
    df_preprocessed = preprocessed_features['df']
    scaler = preprocessed_features['scaler']
    
    # 2. Feature Selection
    context.log.info("Computing correlations and selecting features...")
    corr_dat = compute_correlations(df_preprocessed)
    selected_features = select_top_features(corr_dat, config)
    
    # 3. Training
    context.log.info(f"Training model with {len(selected_features)} features...")
    model, metrics = train_model(df_preprocessed, selected_features, config)
    
    if model is None or metrics is None:
        raise Exception("Model training failed")

    final_features = metrics.get('best_features', selected_features)
    
    # 4. Save Artifacts & Log to MLflow
    paths = setup_paths(timestamp)
    ensure_paths(paths)
    
    importance_df = features_importance(model, df_preprocessed[final_features])
    
    # Log to MLflow
    context.log.info("Logging to MLflow...")
    init_mlflow()
    mlflow_params = {
        'n_estimators': config.n_estimators,
        'max_depth': config.max_depth,
        'min_samples_split': config.min_samples_split,
        'scoring': config.scoring
    }
    
    run_id = log_model_to_mlflow(
        model=model,
        params=mlflow_params,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float, np.float64, np.int64))},
        features=final_features,
        scaler=scaler
    )
    
    context.log.info(f"Model trained and logged to MLflow. Run ID: {run_id}")
    
    save_all_artifacts(scaler, model, importance_df, final_features, metrics, paths, timestamp)
    
    return {"run_id": run_id, "metrics": metrics}
