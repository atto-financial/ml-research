import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from joblib import load
from sklearn.preprocessing import StandardScaler
from app.utils_model import load_and_verify_artifact, validate_features, get_artifact_paths

logger = logging.getLogger(__name__)

def load_model(model_path: Path) -> object:
    """Load a model from the specified path."""
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load(str(model_path))
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

def scale_data(data: pd.DataFrame, scaler: StandardScaler, expected_features: List[str]) -> Optional[pd.DataFrame]:
    """Scale data using the provided scaler, ensuring feature consistency."""
    try:
        valid_features = validate_features(expected_features, data, "Input data features")
        if not valid_features:
            logger.error("No valid features for scaling")
            return None
        data = data[valid_features]
        scaled_data = scaler.transform(data)
        scaled_df = pd.DataFrame(scaled_data, columns=valid_features)
        logger.debug(f"Scaled data shape: {scaled_df.shape}, columns: {list(scaled_df.columns)}")
        return scaled_df
    except Exception as e:
        logger.error(f"Failed to scale data: {str(e)}")
        return None

def make_predictions(model: object, data: pd.DataFrame, threshold: float = 0.75) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Make predictions using the provided model."""
    try:
        if not hasattr(model, 'feature_names_in_'):
            logger.error("Model does not have feature_names_in_ attribute")
            return None, None, None
        model_features = set(model.feature_names_in_)
        data_features = set(data.columns)
        unseen = list(data_features - model_features)
        missing = list(model_features - data_features)
        if unseen:
            logger.warning(f"Unseen features at prediction time: {unseen}")
            data = data.drop(columns=unseen, errors='ignore')
        if missing:
            logger.error(f"Missing features required by model: {missing}")
            return None, None, None
        data = data[list(model.feature_names_in_)]
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]
        predictions_adjusted = (probabilities >= threshold).astype(int)
        logger.info(f"Predictions made: predictions={predictions}, probabilities={probabilities}, adjusted={predictions_adjusted}")
        return predictions, probabilities, predictions_adjusted
    except Exception as e:
        logger.error(f"Error in make_predictions: {str(e)}")
        return None, None, None

def predict_fsk_answers(engineered_data: pd.DataFrame, model_path: str = None, scaler_path: str = None) -> Tuple[Dict, int]:
    """Process engineered data, scale, and make predictions using specified or latest model."""
    try:
        # Load artifacts
        paths = get_artifact_paths(model_dir="save_models", model_path=model_path, scaler_path=scaler_path)
        if not paths['scaler_path'] or not paths['model_path']:
            raise FileNotFoundError("Scaler or model path not found in metadata")

        # Load scaler
        scaler = load_and_verify_artifact(paths['scaler_path'], paths['scaler_checksum']) if paths['scaler_checksum'] else load(paths['scaler_path'])
        if not isinstance(scaler, StandardScaler):
            raise TypeError(f"Invalid scaler object: {type(scaler)}")
        logger.info(f"Scaler loaded: {paths['scaler_path']}")

        # Load model
        model = load_and_verify_artifact(paths['model_path'], paths['model_checksum']) if paths['model_checksum'] else load(paths['model_path'])
        logger.info(f"Model loaded: {paths['model_path']}")

        # Validate features
        expected_features = paths.get('final_features', []) or (model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [])
        if not expected_features:
            raise ValueError("No expected features found in metadata or model")
        valid_features = validate_features(expected_features, engineered_data, "Model features")
        if not valid_features:
            missing = set(expected_features) - set(engineered_data.columns)
            raise ValueError(f"Missing required features: {missing}")

        # Check consistency between final_features and model.feature_names_in_
        if hasattr(model, 'feature_names_in_'):
            model_features = set(model.feature_names_in_)
            metadata_features = set(valid_features)
            if model_features != metadata_features:
                logger.warning(f"Feature mismatch: model expects {model_features}, metadata provides {metadata_features}")
                valid_features = list(model_features)  # Prefer model features

        # Scale data
        scaled_df = scale_data(engineered_data, scaler, valid_features)
        if scaled_df is None:
            raise ValueError("Failed to scale data")

        # Make predictions
        predictions, probabilities, predictions_adjusted = make_predictions(model, scaled_df, threshold=0.75)
        if predictions is None:
            raise ValueError("Failed to make predictions")

        results = {
            'default_probability': round(float(probabilities[0]), 3),
            'model_prediction': int(predictions[0]),
            'adjust_prediction': int(predictions_adjusted[0]),
            'scaler_path': paths['scaler_path'],
            'scaler_checksum': paths['scaler_checksum'],
            'model_path': paths['model_path'],
            'model_checksum': paths['model_checksum'],
            'used_features': list(scaled_df.columns)
        }
        logger.info(f"Prediction results: {results}")
        return results, 200

    except (FileNotFoundError, TypeError, ValueError) as e:
        logger.error(f"Error in predict_fsk_answers: {str(e)}")
        return {"error": str(e)}, 400
    except Exception as e:
        logger.error(f"Internal Server Error in predict_fsk_answers: {str(e)}")
        return {"error": f"Internal Server Error: {str(e)}"}, 500