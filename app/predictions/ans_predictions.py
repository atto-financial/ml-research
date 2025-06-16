import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from joblib import load
from sklearn.preprocessing import StandardScaler
from app.utils_model import load_and_verify_artifact, validate_features, get_artifact_paths

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_model(model_path: Path) -> object:
    
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load(str(model_path))
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}", exc_info=True)
        raise

def scaler_function(cus_engineered_data: pd.DataFrame, scaler: StandardScaler, expected_features: List[str]) -> Optional[pd.DataFrame]:
    if isinstance(expected_features, (np.ndarray, pd.Series)):
        expected_features = expected_features.tolist()
    if expected_features and isinstance(expected_features[0], dict):
        expected_features = [f['feature'] for f in expected_features if 'feature' in f]
    try:
        logger.debug(f"Input data shape: {cus_engineered_data.shape}, columns: {list(cus_engineered_data.columns)}")
        
        if cus_engineered_data.isna().any().any():
            nan_cols = cus_engineered_data.columns[cus_engineered_data.isna().any()].tolist()
            logger.error(f"NaN values found in columns: {nan_cols}")
            return None
        
        if np.isinf(cus_engineered_data).any().any():
            inf_cols = cus_engineered_data.columns[np.isinf(cus_engineered_data).any()].tolist()
            logger.error(f"Infinite values found in columns: {inf_cols}")
            return None

        non_numeric_cols = cus_engineered_data.select_dtypes(exclude=[np.number]).columns
        if non_numeric_cols.any():
            logger.error(f"Non-numeric columns found: {list(non_numeric_cols)}")
            return None
        
        scaler_features = getattr(scaler, 'feature_names_in_', None)
        input_features = set(cus_engineered_data.columns)
        missing_features = set(scaler_features) - input_features
        if missing_features:
            logger.error(f"Input data missing {len(missing_features)} features required by scaler: {missing_features}")
            return None
        extra_features = input_features - set(scaler_features)
        if extra_features:
            logger.warning(f"Input data has {len(extra_features)} extra features: {extra_features}")

        scaler_function = cus_engineered_data[scaler_features]
        logger.debug(f"Data to scale shape: {scaler_function.shape}, columns: {list(scaler_function.columns)}")
        scaled_data = scaler.transform(scaler_function)
        scaled_df = pd.DataFrame(scaled_data, columns=scaler_features, index=cus_engineered_data.index)
        logger.debug(f"Scaled data shape: {scaled_df.shape}, columns: {list(scaled_df.columns)}")

        valid_features = validate_features(expected_features, scaled_df, "Model features")
        if not valid_features:
            missing = set(expected_features) - set(scaled_df.columns)
            logger.error(f"No valid features for model. Missing: {missing}")
            return None
        scaled_df = scaled_df[valid_features]
        logger.info(f"Data scaling and feature selection successful. Final shape: {scaled_df.shape}, columns: {list(scaled_df.columns)}")
        return scaled_df
    except ValueError as ve:
        logger.error(f"ValueError in scaler_function: {str(ve)}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error in scaler_function: {str(e)}", exc_info=True)
        return None

def prediction_function(model: object, scaled_cus_data: pd.DataFrame, adjusted_threshold: float = 0.75) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
   
    try:
        if scaled_cus_data is None or scaled_cus_data.empty:
            logger.error("Scaled data is None or empty")
            return None, None, None

        if not hasattr(model, 'feature_names_in_'):
            logger.error("Model does not have feature_names_in_ attribute")
            return None, None, None

        model_features = list(model.feature_names_in_)
        scaled_cus_features = set(scaled_cus_data.columns)
        missing = set(model_features) - scaled_cus_features
        if missing:
            logger.error(f"Missing {len(missing)} features required by model: {missing}")
            return None, None, None
        extra = scaled_cus_features - set(model_features)
        if extra:
            logger.warning(f"Extra features in scaled data: {extra}")
            scaled_cus_data = scaled_cus_data[model_features]

        scaled_cus_data = scaled_cus_data[model_features]
        logger.debug(f"Prediction input shape: {scaled_cus_data.shape}, columns: {list(scaled_cus_data.columns)}")

        predictions = model.predict(scaled_cus_data)
        if predictions is None or len(predictions) == 0:
            logger.error("Model predict returned None or empty array")
            return None, None, None
        
        probabilities = model.predict_proba(scaled_cus_data)[:, 1]
        if probabilities is None or len(probabilities) == 0:
            logger.error("Model predict_proba returned None or empty array")
            return None, None, None
        
        predictions_adjusted = (probabilities >= adjusted_threshold).astype(int)
        logger.info(f"Predictions made: predictions={predictions.tolist()}, "
                   f"probabilities={probabilities.tolist()}, adjusted={predictions_adjusted.tolist()}")
        return predictions, probabilities, predictions_adjusted
    except ValueError as ve:
        logger.error(f"ValueError in prediction_function: {str(ve)}", exc_info=True)
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error in prediction_function: {str(e)}", exc_info=True)
        return None, None, None

def predict_answers(cus_engineered_data: pd.DataFrame, model_path: str = None, scaler_path: str = None) -> Tuple[Dict, int]:
    try:
        if not isinstance(cus_engineered_data, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(cus_engineered_data)}")
            raise ValueError(f"Input must be a pandas DataFrame, got {type(cus_engineered_data)}")
        if cus_engineered_data.empty:
            logger.error("Input data is empty")
            raise ValueError("Input data is empty")
        logger.debug(f"Input data shape: {cus_engineered_data.shape}, columns: {list(cus_engineered_data.columns)}")

        paths = get_artifact_paths(model_dir="save_models", model_path=model_path, scaler_path=scaler_path)
        if not paths['scaler_path'] or not paths['model_path']:
            logger.error("Scaler or model path not found in metadata")
            raise FileNotFoundError("Scaler or model path not found in metadata")

        scaler = load_and_verify_artifact(paths['scaler_path'], paths['scaler_checksum']) if paths['scaler_checksum'] else load(paths['scaler_path'])
        if not isinstance(scaler, StandardScaler):
            logger.error(f"Invalid scaler object: {type(scaler)}")
            raise TypeError(f"Invalid scaler object: {type(scaler)}")
        logger.info(f"Scaler loaded: {paths['scaler_path']}")

        model = load_and_verify_artifact(paths['model_path'], paths['model_checksum']) if paths['model_checksum'] else load(paths['model_path'])
        logger.info(f"Model loaded: {paths['model_path']}")

        expected_features = paths.get('final_features', []) or (model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [])
        if isinstance(expected_features, (np.ndarray, pd.Series)):
            expected_features = expected_features.tolist()
        if not expected_features:
            logger.error("No expected features found in metadata or model")
            raise ValueError("No expected features found in metadata or model")

        scaled_df = scaler_function(cus_engineered_data, scaler, expected_features)
        if scaled_df is None:
            logger.error("Failed to scale data or select features")
            raise ValueError("Failed to scale data or select features: check input features or scaler")

        if len(scaled_df) != 1:
            logger.warning(f"Expected single-row input, got {len(scaled_df)} rows. Using first row.")
            scaled_df = scaled_df.iloc[[0]]

        prediction_result = prediction_function(model, scaled_df)
        if prediction_result is None or any(x is None for x in prediction_result):
            logger.error("Failed to make predictions: check model compatibility or input features")
            raise ValueError("Failed to make predictions: check model compatibility or input features")
        predictions, probabilities, predictions_adjusted = prediction_result

        results = {
            'default_probability': round(float(probabilities[0]), 3),
            'model_prediction': int(predictions[0]),
            'adjust_prediction': int((probabilities[0])),
            'scaler_path': paths['scaler_path'],
            'scaler_checksum': paths['scaler_checksum'],
            'model_path': paths['model_path'],
            'model_checksum': paths['model_checksum'],
            'used_features': list(scaled_df.columns)
        }
        logger.info(f"Prediction results: default_probability={results['default_probability']}, "
                   f"model_prediction={results['model_prediction']}, adjust_prediction={results['adjust_prediction']}")
        return results, 200

    except (FileNotFoundError, TypeError, ValueError) as e:
        logger.error(f"Validation error in predict_answers: {str(e)}", exc_info=True)
        return {"error": f"Validation error: {str(e)}"}, 400
    except Exception as e:
        logger.error(f"Internal server error in predict_answers: {str(e)}", exc_info=True)
        return {"error": f"Internal server error: {str(e)}"}, 500