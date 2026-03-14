import mlflow
import mlflow.sklearn
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def init_mlflow(experiment_name: str = "Atto_Fintech_Loan_Approval"):
    """Initialize MLflow tracking URI and experiment."""
    # Local tracking by default
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow initialized with experiment: {experiment_name}")

def log_model_to_mlflow(
    model: Any, 
    params: Dict[str, Any], 
    metrics: Dict[str, float], 
    features: list,
    model_name: str = "loan_approval_model",
    scaler: Optional[Any] = None
):
    """Log model, parameters, metrics and artifacts to MLflow."""
    try:
        with mlflow.start_run() as run:
            # Log Parameters
            mlflow.log_params(params)
            
            # Log Metrics
            mlflow.log_metrics(metrics)
            
            # Log Features as a tag or artifact
            mlflow.set_tag("features_count", len(features))
            
            # Log Model
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="model",
                registered_model_name=model_name
            )
            
            # Log Scaler if provided
            if scaler:
                import joblib
                scaler_path = "scaler.pkl"
                joblib.dump(scaler, scaler_path)
                mlflow.log_artifact(scaler_path, artifact_path="preprocessor")
                os.remove(scaler_path)

            logger.info(f"Successfully logged run to MLflow. Run ID: {run.info.run_id}")
            return run.info.run_id
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {str(e)}")
        return None
