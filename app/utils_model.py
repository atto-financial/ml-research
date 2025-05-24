import hashlib
import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from glob import glob

logger = logging.getLogger(__name__)

def calculate_checksum(file_path: str) -> str:
    """Calculate SHA256 checksum of a file."""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        checksum = sha256_hash.hexdigest()
        logger.debug(f"Calculated checksum for {file_path}: {checksum}")
        return checksum
    except Exception as e:
        logger.error(f"Failed to calculate checksum for {file_path}: {str(e)}")
        raise

def save_artifact(obj: Any, directory: str, filename: str, obj_type: str) -> Tuple[str, str]:
    """Save an object (model, scaler, or DataFrame) to disk and return its path and checksum."""
    try:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            dump(obj, file_path)
        checksum = calculate_checksum(file_path)
        logger.info(f"Saved {obj_type} to {file_path} with checksum: {checksum}")
        return file_path, checksum
    except Exception as e:
        logger.error(f"Failed to save {obj_type} to {file_path}: {str(e)}")
        raise

def validate_features(features: List[str], df: pd.DataFrame, name: str) -> List[str]:
    """Validate that a list of features exists in a DataFrame."""
    try:
        if not isinstance(features, list):
            logger.error(f"{name} is not a list: {type(features)}")
            return []
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.error(f"{name} not found in data: {missing}")
            return []
        logger.debug(f"Validated features for {name}: {features}")
        return features
    except Exception as e:
        logger.error(f"Failed to validate features for {name}: {str(e)}")
        return []

def features_importance(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Calculate feature importance for a RandomForestClassifier."""
    try:
        if not isinstance(model, RandomForestClassifier):
            logger.error(f"Model is not a RandomForestClassifier: {type(model)}")
            return pd.DataFrame()
        if not hasattr(model, 'feature_importances_'):
            logger.error("Model lacks feature_importances_ attribute")
            return pd.DataFrame()
        if len(X.columns) != len(model.feature_importances_):
            logger.error(f"Mismatch: {len(X.columns)} columns vs {len(model.feature_importances_)} importances")
            return pd.DataFrame()
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        logger.info(f"Feature Importance:\n{importance_df.to_string()}")
        return importance_df
    except Exception as e:
        logger.error(f"Feature importance calculation failed: {str(e)}")
        return pd.DataFrame()

def load_latest_model_metadata(model_dir: str) -> Dict:
    """Load the latest model metadata from a JSON file."""
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    default_metadata = {
        'cv_roc_auc': 0.0,
        'model_path': '',
        'model_checksum': '',
        'scaler_path': '',
        'scaler_checksum': '',
        'feature_importance': [],
        'cv_accuracy': 0.0,
        'cv_precision': 0.0,
        'cv_recall': 0.0,
        'cv_f1': 0.0,
        'final_features': []
    }
    try:
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")
            return default_metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        for key, default in default_metadata.items():
            metadata.setdefault(key, default)
        if not isinstance(metadata['cv_roc_auc'], (int, float)) or np.isnan(metadata['cv_roc_auc']):
            logger.warning(f"Invalid cv_roc_auc: {metadata['cv_roc_auc']}. Using 0.0")
            metadata['cv_roc_auc'] = 0.0
        logger.info(f"Loaded metadata from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {str(e)}")
        return default_metadata

def save_model_metadata(model_dir: str, cv_roc_auc: float, model_path: str, model_checksum: str, 
                        scaler_path: str, scaler_checksum: str, feature_importance: pd.DataFrame, 
                        metrics: Dict, filename: str = "model_metadata.json") -> None:
    """Save model metadata to a JSON file."""
    metadata_path = os.path.join(model_dir, filename)
    try:
        scaler_path = scaler_path if scaler_path and os.path.exists(scaler_path) else ""
        scaler_checksum = scaler_checksum if scaler_checksum and len(scaler_checksum) == 64 else ""
        feature_importance_dict = feature_importance.to_dict('records') if not feature_importance.empty else []
        if feature_importance.empty:
            logger.warning("Feature importance is empty")
        
        metric_values = {
            'cv_accuracy': float(metrics.get('cv_accuracy', 0.0)),
            'cv_precision': float(metrics.get('cv_precision', 0.0)),
            'cv_recall': float(metrics.get('cv_recall', 0.0)),
            'cv_f1': float(metrics.get('cv_f1', 0.0)),
            'final_features': metrics.get('final_features', [])
        }
        for m, value in metric_values.items():
            if m != 'final_features' and (not isinstance(value, (int, float)) or np.isnan(value) or value < 0.0 or value > 1.0):
                logger.warning(f"Invalid {m}: {value}. Setting to 0.0")
                metric_values[m] = 0.0

        metadata = {
            'cv_roc_auc': float(cv_roc_auc),
            'model_path': model_path,
            'model_checksum': model_checksum,
            'scaler_path': scaler_path,
            'scaler_checksum': scaler_checksum,
            'feature_importance': feature_importance_dict,
            **metric_values
        }
        os.makedirs(model_dir, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata to {metadata_path}: {str(e)}")
        raise

def load_and_verify_artifact(file_path: str, expected_checksum: str) -> Any:
    """Load an artifact and verify its checksum."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact file not found: {file_path}")
        current_checksum = calculate_checksum(file_path)
        if expected_checksum and current_checksum != expected_checksum:
            raise ValueError(f"Checksum mismatch for {file_path}: expected {expected_checksum}, got {current_checksum}")
        obj = load(file_path)
        logger.info(f"Loaded and verified {file_path} with checksum: {current_checksum}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load and verify artifact {file_path}: {str(e)}")
        raise

def get_artifact_paths(model_dir: str, model_path: str = None, scaler_path: str = None) -> Dict:
    """Retrieve artifact paths and checksums, prioritizing specified paths or selecting best model by cv_roc_auc."""
    default_paths = {
        'scaler_path': '',
        'scaler_checksum': '',
        'model_path': '',
        'model_checksum': '',
        'final_features': []
    }

    # If specific paths are provided, use them
    if model_path and scaler_path:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Specified model path not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Specified scaler path not found: {scaler_path}")
            model_checksum = calculate_checksum(model_path) if os.path.exists(model_path) else ''
            scaler_checksum = calculate_checksum(scaler_path) if os.path.exists(scaler_path) else ''
            metadata_path = os.path.join(model_dir, "model_metadata.json")
            final_features = []
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                final_features = metadata.get('final_features', [])
            paths = {
                'scaler_path': scaler_path,
                'scaler_checksum': scaler_checksum,
                'model_path': model_path,
                'model_checksum': model_checksum,
                'final_features': final_features
            }
            logger.info(f"Using specified artifact paths: {paths}")
            return paths
        except Exception as e:
            logger.error(f"Failed to use specified artifact paths: {str(e)}")
            return default_paths

    # Otherwise, find the best model by cv_roc_auc
    try:
        metadata_files = glob(os.path.join(model_dir, "model_metadata_*.json"))
        if not metadata_files:
            logger.warning(f"No metadata files found in {model_dir}")
            return default_paths

        best_metadata = None
        best_cv_roc_auc = -float('inf')
        for metadata_path in metadata_files:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                cv_roc_auc = float(metadata.get('cv_roc_auc', 0.0))
                if cv_roc_auc > best_cv_roc_auc and os.path.exists(metadata.get('model_path', '')) and os.path.exists(metadata.get('scaler_path', '')):
                    best_cv_roc_auc = cv_roc_auc
                    best_metadata = metadata
            except Exception as e:
                logger.warning(f"Failed to read metadata {metadata_path}: {str(e)}")
                continue

        if not best_metadata:
            logger.error(f"No valid metadata found in {model_dir}")
            return default_paths

        paths = {
            'scaler_path': best_metadata.get('scaler_path', ''),
            'scaler_checksum': best_metadata.get('scaler_checksum', ''),
            'model_path': best_metadata.get('model_path', ''),
            'model_checksum': best_metadata.get('model_checksum', ''),
            'final_features': best_metadata.get('final_features', [])
        }

        for key in ['scaler_path', 'model_path']:
            if paths[key] and not os.path.exists(paths[key]):
                logger.error(f"{key} does not exist: {paths[key]}")
                paths[key] = ''
                paths[key.replace('_path', '_checksum')] = ''

        logger.info(f"Selected best model with cv_roc_auc={best_cv_roc_auc}: {paths}")
        return paths
    except Exception as e:
        logger.error(f"Failed to load artifact paths from {model_dir}: {str(e)}")
        return default_paths