import hashlib
import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sklearn
import joblib

logger = logging.getLogger(__name__)

def setup_paths(timestamp: str) -> Dict[str, Tuple[str, str]]:
   
    return {
        'scaler': ("save_scalers", f"custom_scaler_{timestamp}.pkl"),
        'model': ("save_models", f"custom_model_{timestamp}.pkl"),
        'output': ("output_data", f"feature_importance_{timestamp}.csv")
    }

def calculate_checksum(file_path: str) -> str:
   
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        checksum = sha256_hash.hexdigest()
        logger.debug(f"Calculated checksum for {file_path}: {checksum}")
        return checksum
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Failed to calculate checksum for {file_path}: {str(e)}")
        raise

def validate_data(df: Optional[pd.DataFrame], name: str, min_rows: int = 1, min_cols: int = 1) -> bool:
    
    try:
        if df is None:
            logger.error(f"{name} is None")
            return False
        if not isinstance(df, pd.DataFrame):
            logger.error(f"{name} is not a DataFrame: {type(df)}")
            return False
        if df.empty:
            logger.error(f"{name} is empty")
            return False
        if len(df) < min_rows:
            logger.error(f"{name} has too few rows: {len(df)} < {min_rows}")
            return False
        if len(df.columns) < min_cols:
            logger.error(f"{name} has too few columns: {len(df.columns)} < {min_cols}")
            return False
        if df.isna().all().all():
            logger.error(f"{name} contains only NaN values")
            return False
        logger.debug(f"Validated {name}: {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        logger.error(f"Failed to validate {name}: {str(e)}")
        return False

def validate_features(features: List[str], df: pd.DataFrame, name: str) -> List[str]:
    
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

def save_artifact(obj: Any, directory: str, filename: str, obj_type: str) -> Tuple[str, str]:
   
    try:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
        
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(file_path, index=False, encoding='utf-8-sig')
        elif isinstance(obj, (StandardScaler, RandomForestClassifier)):
            dump(obj, file_path)
        else:
            raise ValueError(f"Unsupported object type for saving: {type(obj)}")
        
        checksum = calculate_checksum(file_path)
        logger.info(f"Saved {obj_type} to {file_path} with checksum: {checksum}")
        return file_path, checksum
    except (OSError, PermissionError, ValueError) as e:
        logger.error(f"Failed to save {obj_type} to {file_path}: {str(e)}")
        raise

def load_and_verify_artifact(file_path: str, expected_checksum: str, expected_type: type = None) -> Any:
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact file not found: {file_path}")
        current_checksum = calculate_checksum(file_path)
        if expected_checksum and current_checksum != expected_checksum:
            raise ValueError(f"Checksum mismatch for {file_path}: expected {expected_checksum}, got {current_checksum}")
        
        obj = load(file_path)
        if expected_type and not isinstance(obj, expected_type):
            raise ValueError(f"Loaded object is not of expected type {expected_type}: got {type(obj)}")
        
        logger.info(f"Loaded and verified {file_path} with checksum: {current_checksum}")
        return obj
    except (FileNotFoundError, ValueError, PermissionError) as e:
        logger.error(f"Failed to load and verify artifact {file_path}: {str(e)}")
        raise

def save_model_metadata(model_dir: str, metadata: Dict, filename: str = "model_metadata.json") -> None:
    
    metadata_path = os.path.join(model_dir, filename)
    try:
        metadata['package_versions'] = {
            'sklearn': sklearn.__version__,
            'joblib': joblib.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__
        }
        
        for key in ['model_path', 'scaler_path', 'feature_importance_path']:
            path = metadata.get(key, '')
            checksum = metadata.get(key.replace('_path', '_checksum'), '')
            if path and not os.path.exists(path):
                logger.warning(f"{key} does not exist: {path}")
                metadata[key] = ''
                metadata[key.replace('_path', '_checksum')] = ''
            if checksum and len(checksum) != 64:
                logger.warning(f"Invalid checksum for {key}: {checksum}")
                metadata[key.replace('_path', '_checksum')] = ''
        
        for metric in ['cv_roc_auc', 'cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1']:
            value = metadata.get(metric, 0.0)
            if not isinstance(value, (int, float)) or np.isnan(value) or value < 0.0:
                logger.warning(f"Invalid {metric}: {value}. Setting to 0.0")
                metadata[metric] = 0.0
        
        if not isinstance(metadata.get('final_features', []), list):
            logger.warning(f"Invalid final_features: {metadata['final_features']}. Setting to []")
            metadata['final_features'] = []
        
        os.makedirs(model_dir, exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_path}")
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to save metadata to {metadata_path}: {str(e)}")
        raise

def load_latest_model_metadata(model_dir: str) -> Dict:
    
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    default_metadata = {
        'cv_roc_auc': 0.0,
        'model_path': '',
        'model_checksum': '',
        'scaler_path': '',
        'scaler_checksum': '',
        'feature_importance_path': '',
        'feature_importance_checksum': '',
        'feature_importance': [],
        'cv_accuracy': 0.0,
        'cv_precision': 0.0,
        'cv_recall': 0.0,
        'cv_f1': 0.0,
        'final_features': [],
        'timestamp': '',
        'package_versions': {}
    }
    try:
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")
            return default_metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        for key, default in default_metadata.items():
            metadata.setdefault(key, default)
        for metric in ['cv_roc_auc', 'cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1']:
            value = metadata.get(metric, 0.0)
            if not isinstance(value, (int, float)) or np.isnan(value) or value < 0.0:
                logger.warning(f"Invalid {metric}: {value}. Using 0.0")
                metadata[metric] = 0.0
        if not isinstance(metadata.get('final_features', []), list):
            logger.warning(f"Invalid final_features: {metadata['final_features']}. Using []")
            metadata['final_features'] = []
        logger.info(f"Loaded metadata from {metadata_path}")
        return metadata
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {str(e)}")
        return default_metadata

def get_artifact_paths(model_dir: str, model_path: str = None, scaler_path: str = None) -> Dict:
    
    default_paths = {
        'scaler_path': '',
        'scaler_checksum': '',
        'model_path': '',
        'model_checksum': '',
        'feature_importance_path': '',
        'feature_importance_checksum': '',
        'final_features': []
    }
    try:
        if model_path and scaler_path:
            model_checksum = calculate_checksum(model_path) if os.path.exists(model_path) else ''
            scaler_checksum = calculate_checksum(scaler_path) if os.path.exists(scaler_path) else ''
            metadata = load_latest_model_metadata(model_dir)
            paths = {
                'scaler_path': scaler_path,
                'scaler_checksum': scaler_checksum,
                'model_path': model_path,
                'model_checksum': model_checksum,
                'feature_importance_path': metadata.get('feature_importance_path', ''),
                'feature_importance_checksum': metadata.get('feature_importance_checksum', ''),
                'final_features': metadata.get('final_features', [])
            }
            logger.info(f"Using specified artifact paths: {paths}")
            return paths
        metadata = load_latest_model_metadata(model_dir)
        paths = {
            'scaler_path': metadata.get('scaler_path', ''),
            'scaler_checksum': metadata.get('scaler_checksum', ''),
            'model_path': metadata.get('model_path', ''),
            'model_checksum': metadata.get('model_checksum', ''),
            'feature_importance_path': metadata.get('feature_importance_path', ''),
            'feature_importance_checksum': metadata.get('feature_importance_checksum', ''),
            'final_features': metadata.get('final_features', [])
        }
        for key in ['scaler_path', 'model_path', 'feature_importance_path']:
            if paths[key] and not os.path.exists(paths[key]):
                logger.warning(f"{key} does not exist: {paths[key]}")
                paths[key] = ''
                paths[key.replace('_path', '_checksum')] = ''
        logger.info(f"Retrieved artifact paths: {paths}")
        return paths
    except Exception as e:
        logger.error(f"Failed to retrieve artifact paths from {model_dir}: {str(e)}")
        return default_paths

def ensure_paths(paths: Dict[str, Tuple[str, str]]) -> Tuple[bool, str]:
   
    for path_type, (dir_path, _) in paths.items():
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create directory {dir_path}: {str(e)}")
            return False, str(e)
    return True, ""

def save_all_artifacts(scaler: Any, model: RandomForestClassifier, importance_df: pd.DataFrame, 
                      final_features: List[str], metrics: Dict, paths: Dict[str, Tuple[str, str]], 
                      timestamp: str) -> Optional[Dict[str, str]]:
   
    try:
        if not isinstance(scaler, StandardScaler):
            logger.error(f"Scaler is not a StandardScaler: {type(scaler)}")
            return None
        
        scaler_path, scaler_checksum = save_artifact(scaler, *paths['scaler'], "scaler")
        model_path, model_checksum = save_artifact(model, *paths['model'], "model")
        importance_path, importance_checksum = "", ""
        
        if not importance_df.empty:
            importance_path, importance_checksum = save_artifact(importance_df, *paths['output'], "feature_importance")
            logger.info(f"Feature importance saved at {importance_path} with checksum {importance_checksum}")
        else:
            logger.warning("Feature importance is empty.")
        
        metadata = {
            'cv_roc_auc': float(metrics.get('cross_validated_roc_auc', 0.0)),
            'cv_accuracy': float(metrics.get('cross_validated_accuracy', 0.0)),
            'cv_precision': float(metrics.get('cross_validated_precision', 0.0)),
            'cv_recall': float(metrics.get('cross_validated_recall', 0.0)),
            'cv_f1': float(metrics.get('cross_validated_f1', 0.0)),
            'final_features': final_features,
            'model_path': model_path,
            'model_checksum': model_checksum,
            'scaler_path': scaler_path,
            'scaler_checksum': scaler_checksum,
            'feature_importance_path': importance_path,
            'feature_importance_checksum': importance_checksum,
            'feature_importance': importance_df.to_dict('records') if not importance_df.empty else [],
            'timestamp': timestamp
        }
        
        save_model_metadata(paths['model'][0], metadata)
        return {
            'scaler_path': scaler_path,
            'scaler_checksum': scaler_checksum,
            'model_path': model_path,
            'model_checksum': model_checksum,
            'importance_path': importance_path,
            'importance_checksum': importance_checksum
        }
    except (OSError, PermissionError, ValueError) as e:
        logger.error(f"Failed to save artifacts: {str(e)}")
        return None

def check_scaler_type(scaler_path: str, model_dir: str = "save_models") -> Dict:
    
    try:
        
        paths = get_artifact_paths(model_dir=model_dir, scaler_path=scaler_path)
        scaler_checksum = paths.get('scaler_checksum', None)

        scaler = load_and_verify_artifact(
            file_path=scaler_path,
            expected_checksum=scaler_checksum,
            expected_type=StandardScaler
        )

        result = {
            'status': 'success',
            'scaler_type': str(type(scaler)),
            'is_standard_scaler': isinstance(scaler, StandardScaler),
            'message': f"Scaler loaded successfully from {scaler_path}",
            'checksum_verified': scaler_checksum is not None
        }
        logger.info(f"Scaler check result: {result}")
        return result

    except FileNotFoundError:
        result = {
            'status': 'error',
            'scaler_type': 'None',
            'is_standard_scaler': False,
            'message': f"Scaler file not found: {scaler_path}",
            'checksum_verified': False
        }
        logger.error(result['message'])
        return result
    except ValueError as ve:
        result = {
            'status': 'error',
            'scaler_type': 'Unknown',
            'is_standard_scaler': False,
            'message': f"Invalid scaler: {str(ve)}",
            'checksum_verified': scaler_checksum is not None
        }
        logger.error(result['message'])
        return result
    except Exception as e:
        result = {
            'status': 'error',
            'scaler_type': 'Unknown',
            'is_standard_scaler': False,
            'message': f"Failed to check scaler type: {str(e)}",
            'checksum_verified': False
        }
        logger.error(result['message'])
        return result