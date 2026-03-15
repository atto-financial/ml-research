from dagster import asset, AssetExecutionContext, MetadataValue
from datetime import datetime
import os

@asset(group_name="health_check")
def db_health_check(context: AssetExecutionContext) -> bool:
    """
    Lightweight DB connectivity check.
    Materialize this asset to verify the PostgreSQL connection is working.
    """
    from app.utils.db_connection import get_db_connection
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        context.log.info("✅ DB connection successful")
        context.add_output_metadata({"status": MetadataValue.text("✅ Connected")})
        return True
    except Exception as e:
        context.log.error(f"❌ DB connection failed: {e}")
        raise

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
def fsk_local_cache(context: AssetExecutionContext) -> bool:
    """
    Sync PostgreSQL data to local DuckDB cache to save resources.
    """
    from app.utils.duckdb_utils import sync_to_duckdb
    sql_path = os.path.join(os.path.dirname(__file__), "../app/data/queries/load_loan_data.sql")
    with open(sql_path) as f:
        query = f.read()
    success = sync_to_duckdb("fct_loan_features", query)
    if success:
        context.log.info("Successfully cached data to DuckDB")
    return success

@asset(group_name="data_pipeline")
def fsk_engineered_data(context: AssetExecutionContext, fsk_local_cache: bool):
    """
    Load data from local DuckDB cache.
    """
    import pandas as pd
    from app.utils.duckdb_utils import query_duckdb
    from app.data.data_loading import data_loading_fsk_v1
    if not fsk_local_cache:
        context.log.warning("Cache failed, falling back to PostgreSQL")
        return data_loading_fsk_v1()
        
    data = query_duckdb("SELECT * FROM fct_loan_features")
    context.log.info(f"Loaded {len(data)} rows from DuckDB")
    return data

@asset(group_name="ml_pipeline")
def preprocessed_features(context: AssetExecutionContext, fsk_engineered_data):
    """
    Use Hamilton to modularly preprocess features.
    """
    from hamilton import driver
    from app.models import features_hamilton
    
    adapter = driver.Driver({}, features_hamilton)
    
    # Execute separately: final_preprocessed_data returns a DataFrame directly
    df_preprocessed = adapter.execute(
        ['final_preprocessed_data'],
        inputs={'df': fsk_engineered_data}
    )
    
    # feature_scaler is a non-DataFrame object, Hamilton wraps it in a column
    scaler = adapter.execute(
        ['feature_scaler'],
        inputs={'df': fsk_engineered_data}
    )['feature_scaler'].iloc[0]
    
    # Drop non-feature columns before passing to model training
    non_feature_cols = ['user_id']
    cols_to_drop = [c for c in non_feature_cols if c in df_preprocessed.columns]
    if cols_to_drop:
        context.log.info(f"Dropping non-feature columns: {cols_to_drop}")
        df_preprocessed = df_preprocessed.drop(columns=cols_to_drop)
    
    context.log.info(f"Preprocessing complete. Columns: {len(df_preprocessed.columns)}")
    return {"df": df_preprocessed, "scaler": scaler}

@asset(group_name="ml_pipeline")
def lucis_model(context: AssetExecutionContext, preprocessed_features: dict):
    """
    Train Lucis model (Random Forest) using preprocessed data.
    """
    import numpy as np
    from app.models.lucis import train_model, select_top_features, ModelConfig
    from app.models.Utils_statistics import compute_correlations
    from app.utils.utils_mlflow import init_mlflow, log_model_to_mlflow
    from app.utils.utils_model import setup_paths, ensure_paths, save_all_artifacts, features_importance
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = ModelConfig()
    
    df_preprocessed = preprocessed_features['df']
    scaler = preprocessed_features['scaler']
    
    # Drop non-feature columns (defensive: handles old cached data too)
    non_feature_cols = ['user_id']
    cols_to_drop = [c for c in non_feature_cols if c in df_preprocessed.columns]
    if cols_to_drop:
        context.log.info(f"Dropping non-feature columns from training data: {cols_to_drop}")
        df_preprocessed = df_preprocessed.drop(columns=cols_to_drop)
    
    context.log.info(f"Data shape: {df_preprocessed.shape}, columns: {list(df_preprocessed.columns)}")
    
    # 2. Feature Selection
    context.log.info("Computing correlations and selecting features...")
    corr_dat = compute_correlations(df_preprocessed)
    selected_features = select_top_features(corr_dat, config)
    
    if not selected_features:
        raise Exception("Feature selection returned no features. Check correlation data.")
    
    # 3. Training
    context.log.info(f"Training model with {len(selected_features)} features: {selected_features}")
    model, metrics = train_model(df_preprocessed, selected_features, config)
    
    if model is None or metrics is None:
        raise Exception(
            f"Model training failed. Possible causes: "
            f"target column 'ust' missing ({('ust' in df_preprocessed.columns)}), "
            f"data shape={df_preprocessed.shape}, "
            f"selected_features={len(selected_features)}"
        )

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
