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
        'metadata': ("output_data", f"metadata_{timestamp}.csv")
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

def validate_features(features: List[Any], df: pd.DataFrame, name: str) -> List[Any]:
    try:
        if not isinstance(features, list):
            logger.error(f"{name} is not a list: {type(features)}")
            return []
        
        if not features:
            logger.error(f"{name} is empty")
            return []
        
        if all(isinstance(item, str) for item in features):
            missing = [f for f in features if f not in df.columns]
            if missing:
                logger.error(f"{name} not found in data: {missing}")
                return []
            logger.debug(f"Validated features for {name}: {features}")
            return features
        
        if all(isinstance(item, dict) for item in features):
            for item in features:
                if 'feature' not in item or 'importance' not in item:
                    logger.error(f"Invalid feature format in {name}: {item}")
                    return []
                if item['feature'] not in df.columns:
                    logger.error(f"Feature {item['feature']} not found in data")
                    return []
                if not isinstance(item['importance'], (int, float)) or item['importance'] < 0:
                    logger.error(f"Invalid importance for {item['feature']}: {item['importance']}")
                    return []
            logger.debug(f"Validated features for {name}: {[item['feature'] for item in features]}")
            return features
        
        logger.error(f"Mixed or invalid feature formats in {name}: {features}")
        return []
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
        
        for key in ['model_path', 'scaler_path', 'metadata_path']:
            path = metadata.get(key, '')
            checksum = metadata.get(key.replace('_path', '_checksum'), '')
            if path and not os.path.exists(path):
                logger.warning(f"{key} does not exist: {path}")
                metadata[key] = ''
                metadata[key.replace('_path', '_checksum')] = ''
            if checksum and len(checksum) != 64:
                logger.warning(f"Invalid checksum for {key}: {checksum}")
                metadata[key.replace('_path', '_checksum')] = ''
        
        # Validate metrics
        metrics_keys = [
            'cv_roc_auc', 'cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1',
            'cv_score_mean', 'cv_score_std', 'test_roc_auc', 'test_accuracy',
            'test_precision', 'test_recall', 'test_f1', 'overfitting_gap',
            'accuracy_ci_lower', 'accuracy_ci_upper', 'precision_ci_lower',
            'precision_ci_upper', 'recall_ci_lower', 'recall_ci_upper',
            'f1_ci_lower', 'f1_ci_upper', 'roc_auc_ci_lower', 'roc_auc_ci_upper'
        ]
        for metric in metrics_keys:
            value = metadata.get(metric, 0.0)
            if not isinstance(value, (int, float)) or np.isnan(value) or value < 0.0:
                logger.warning(f"Invalid {metric}: {value}. Setting to 0.0")
                metadata[metric] = 0.0
        
        if not isinstance(metadata.get('final_features', []), list):
            logger.warning(f"Invalid final_features: {metadata['final_features']}. Setting to []")
            metadata['final_features'] = []
        for item in metadata['final_features']:
            if not isinstance(item, dict) or 'feature' not in item or 'importance' not in item:
                logger.warning(f"Invalid feature format: {item}. Setting final_features to []")
                metadata['final_features'] = []
                break
            if not isinstance(item['importance'], (int, float)) or item['importance'] < 0:
                logger.warning(f"Invalid importance for {item['feature']}: {item['importance']}. Setting final_features to []")
                metadata['final_features'] = []
                break
        
        os.makedirs(model_dir, exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_path}")
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to save metadata to {metadata_path}: {str(e)}")
        raise

def save_metadata_csv(metadata: Dict, directory: str, filename: str) -> Tuple[str, str]:
    try:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
        
        if not os.access(directory, os.W_OK):
            logger.error(f"No write permission for directory: {directory}")
            raise PermissionError(f"No write permission for directory: {directory}")
        
        metadata_flat = metadata.copy()
        metadata_flat['final_features'] = json.dumps(metadata_flat['final_features'])
        metadata_flat['package_versions'] = json.dumps(metadata_flat['package_versions'])
        
        metadata_df = pd.DataFrame([metadata_flat])
        metadata_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        if not os.path.exists(file_path):
            logger.error(f"Metadata CSV file was not created: {file_path}")
            raise OSError(f"Metadata CSV file was not created: {file_path}")
        
        checksum = calculate_checksum(file_path)
        logger.info(f"Saved metadata CSV to {file_path} with checksum: {checksum}")
        return file_path, checksum
    except (OSError, PermissionError, ValueError) as e:
        logger.error(f"Failed to save metadata CSV to {file_path}: {str(e)}")
        return "", ""

def load_latest_model_metadata(model_dir: str) -> Dict:
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    default_metadata = {
        'cv_roc_auc': 0.0,
        'cv_accuracy': 0.0,
        'cv_precision': 0.0,
        'cv_recall': 0.0,
        'cv_f1': 0.0,
        'cv_score_mean': 0.0,
        'cv_score_std': 0.0,
        'test_roc_auc': 0.0,
        'test_accuracy': 0.0,
        'test_precision': 0.0,
        'test_recall': 0.0,
        'test_f1': 0.0,
        'overfitting_gap': 0.0,
        'accuracy_ci_lower': 0.0,
        'accuracy_ci_upper': 0.0,
        'precision_ci_lower': 0.0,
        'precision_ci_upper': 0.0,
        'recall_ci_lower': 0.0,
        'recall_ci_upper': 0.0,
        'f1_ci_lower': 0.0,
        'f1_ci_upper': 0.0,
        'roc_auc_ci_lower': 0.0,
        'roc_auc_ci_upper': 0.0,
        'model_path': '',
        'model_checksum': '',
        'scaler_path': '',
        'scaler_checksum': '',
        'metadata_path': '',
        'metadata_checksum': '',
        'feature_importance': [],
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
        for metric in default_metadata:
            if metric not in ['model_path', 'model_checksum', 'scaler_path', 'scaler_checksum',
                              'metadata_path', 'metadata_checksum','final_features', 'timestamp', 'package_versions']:
                value = metadata.get(metric, 0.0)
                if not isinstance(value, (int, float)) or np.isnan(value) or value < 0.0:
                    logger.warning(f"Invalid {metric}: {value}. Using 0.0")
                    metadata[metric] = 0.0
        if not isinstance(metadata.get('final_features', []), list):
            logger.warning(f"Invalid final_features: {metadata['final_features']}. Using []")
            metadata['final_features'] = []
        for item in metadata['final_features']:
            if not isinstance(item, dict) or 'feature' not in item or 'importance' not in item:
                logger.warning(f"Invalid feature format: {item}. Using []")
                metadata['final_features'] = []
                break
            if not isinstance(item['importance'], (int, float)) or item['importance'] < 0:
                logger.warning(f"Invalid importance for {item['feature']}: {item['importance']}. Using []")
                metadata['final_features'] = []
                break
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
        'metadata_path': '',
        'metadata_checksum': '',
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
                'metadata_path': metadata.get('metadata_path', ''),
                'metadata_checksum': metadata.get('metadata_checksum', ''),
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
            'metadata_path': metadata.get('metadata_path', ''),
            'metadata_checksum': metadata.get('metadata_checksum', ''),
            'final_features': metadata.get('final_features', [])
        }
        for key in ['scaler_path', 'model_path', 'feature_importance_path', 'metadata_path']:
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
        metadata_path, metadata_checksum = "", ""

        final_features_dict = [
            {'feature': row['feature'], 'importance': float(row['importance'])}
            for _, row in importance_df.iterrows()
        ]
        
        metadata = {
            'cv_roc_auc': float(metrics.get('cv_roc_auc', 0.0)),
            'cv_accuracy': float(metrics.get('cv_accuracy', 0.0)),
            'cv_precision': float(metrics.get('cv_precision', 0.0)),
            'cv_recall': float(metrics.get('cv_recall', 0.0)),
            'cv_f1': float(metrics.get('cv_f1', 0.0)),
            'cv_score_mean': float(metrics.get('cv_score_mean', 0.0)),
            'cv_score_std': float(metrics.get('cv_score_std', 0.0)),
            'test_roc_auc': float(metrics.get('test_roc_auc', 0.0)),
            'test_accuracy': float(metrics.get('test_accuracy', 0.0)),
            'test_precision': float(metrics.get('test_precision', 0.0)),
            'test_recall': float(metrics.get('test_recall', 0.0)),
            'test_f1': float(metrics.get('test_f1', 0.0)),
            'overfitting_gap': float(metrics.get('overfitting_gap', 0.0)),
            'accuracy_ci_lower': float(metrics.get('accuracy_ci_lower', 0.0)),
            'accuracy_ci_upper': float(metrics.get('accuracy_ci_upper', 0.0)),
            'precision_ci_lower': float(metrics.get('precision_ci_lower', 0.0)),
            'precision_ci_upper': float(metrics.get('precision_ci_upper', 0.0)),
            'recall_ci_lower': float(metrics.get('recall_ci_lower', 0.0)),
            'recall_ci_upper': float(metrics.get('recall_ci_upper', 0.0)),
            'f1_ci_lower': float(metrics.get('f1_ci_lower', 0.0)),
            'f1_ci_upper': float(metrics.get('f1_ci_upper', 0.0)),
            'roc_auc_ci_lower': float(metrics.get('roc_auc_ci_lower', 0.0)),
            'roc_auc_ci_upper': float(metrics.get('roc_auc_ci_upper', 0.0)),
            'final_features': final_features_dict,
            'model_path': model_path,
            'model_checksum': model_checksum,
            'scaler_path': scaler_path,
            'scaler_checksum': scaler_checksum,
            'metadata_path': metadata_path,
            'metadata_checksum': metadata_checksum,
            'timestamp': timestamp
        }
        
        logger.debug(f"Metadata metrics before saving: {metadata['cv_roc_auc']=}, {metadata['cv_accuracy']=}, {metadata['cv_precision']=}, {metadata['cv_recall']=}, {metadata['cv_f1']=}, {metadata['cv_score_mean']=}, {metadata['cv_score_std']=}, {metadata['test_roc_auc']=}, {metadata['test_accuracy']=}, {metadata['test_precision']=}, {metadata['test_recall']=}, {metadata['test_f1']=}, {metadata['overfitting_gap']=}")
        
        logger.debug("Saving initial metadata.json")
        save_model_metadata(paths['model'][0], metadata)
        
        logger.debug("Attempting to save metadata CSV")
        metadata_path, metadata_checksum = save_metadata_csv(metadata, *paths['metadata'])
        if not metadata_path or not metadata_checksum:
            logger.error("Failed to save metadata CSV, proceeding with empty metadata_path and metadata_checksum")
        else:
            metadata['metadata_path'] = metadata_path
            metadata['metadata_checksum'] = metadata_checksum
            logger.debug("Saving updated metadata.json with metadata_path and metadata_checksum")
            save_model_metadata(paths['model'][0], metadata)
        
        return {
            'scaler_path': scaler_path,
            'scaler_checksum': scaler_checksum,
            'model_path': model_path,
            'model_checksum': model_checksum,
            'metadata_path': metadata_path,
            'metadata_checksum': metadata_checksum
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