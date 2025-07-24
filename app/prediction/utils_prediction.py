import logging
import math
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.preprocessing import StandardScaler
from app.utils_model import validate_features, get_artifact_paths, load_and_verify_artifact

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def screen_fsk_answers(answers: Dict[str, List[str]]) -> Tuple[bool, Optional[str]]:

    if not answers:
        logger.error("Empty answers dictionary")
        return False, "Empty answers dictionary"

    all_answers = []
    for key, vals in answers.items():
        if not isinstance(vals, list) or not vals:
            logger.error(f"Invalid or empty list for {key}")
            return False, f"Invalid or empty list for {key}"
        
        try:
            vals = [int(v) for v in vals]
        except ValueError:
            logger.error(f"Non-integer values in {key}")
            return False, f"Non-integer values in {key}"
        
        if not all(1 <= v <= 3 for v in vals):
            logger.error(f"Values in {key} must be between 1 and 3")
            return False, f"Values in {key} must be between 1 and 3"
        
        all_answers.extend(vals)

    n = len(all_answers)
    if n == 0:
        logger.error("No valid answers found")
        return False, "No valid answers found"

    if len(set(all_answers)) == 1:
        logger.warning("All answers identical")
        return False, "All answers identical"

    logger.info("Answers passed screening")
    return True, None

def load_model(model_path: Path, expected_checksum: Optional[str] = None) -> object:
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if expected_checksum:
            model = load_and_verify_artifact(str(model_path), expected_checksum)
        else:
            model = load(str(model_path))
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}", exc_info=True)
        raise

def load_scaler(scaler_path: Path, expected_checksum: Optional[str] = None) -> StandardScaler:
    try:
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if expected_checksum:
            scaler = load_and_verify_artifact(str(scaler_path), expected_checksum, expected_type=StandardScaler)
        else:
            scaler = load(str(scaler_path))
            if not isinstance(scaler, StandardScaler):
                raise TypeError(f"Loaded scaler is not a StandardScaler: {type(scaler)}")
        logger.info(f"Loaded scaler from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Failed to load scaler from {scaler_path}: {str(e)}", exc_info=True)
        raise

def scaler_function(cus_engineered_data: pd.DataFrame, scaler: StandardScaler, expected_features: List[str]) -> Optional[pd.DataFrame]:
    if isinstance(expected_features, (np.ndarray, pd.Series)):
        expected_features = expected_features.tolist()
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
        if len(non_numeric_cols) > 0:
            logger.error(f"Non-numeric columns found: {list(non_numeric_cols)}")
            return None
        
        scaler_features = getattr(scaler, 'feature_names_in_', None)
        if scaler_features is None:
            logger.error("Scaler does not have 'feature_names_in_' attribute")
            return None
        
        input_features = set(cus_engineered_data.columns)
        missing_features = set(scaler_features) - input_features
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features for scaler: {missing_features}. Filling with mean values.")
            for feature in missing_features:
                mean_val = scaler.mean_[list(scaler_features).index(feature)]
                cus_engineered_data[feature] = mean_val  # Fill with scaler's mean for that feature
        
        extra_features = input_features - set(scaler_features)
        if extra_features:
            logger.warning(f"Input data has {len(extra_features)} extra features: {extra_features}")

        data_to_scale = cus_engineered_data[scaler_features]
        logger.debug(f"Data to scale shape: {data_to_scale.shape}, columns: {list(data_to_scale.columns)}")
        scaled_data = scaler.transform(data_to_scale)
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

def prediction_function(
    model: object,
    cus_data: pd.DataFrame,
    scale_data: bool = False,
    scaler: Optional[StandardScaler] = None,
    expected_features: Optional[List[str]] = None,
    adjusted_threshold: float = 0.75
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        if cus_data is None or cus_data.empty:
            logger.error("Data is None or empty")
            return None, None, None

        if not hasattr(model, 'feature_names_in_'):
            logger.error("Model does not have feature_names_in_ attribute")
            return None, None, None

        model_features = list(model.feature_names_in_)

        if scale_data:
            if scaler is None:
                logger.error("Scaler is required when scale_data=True")
                return None, None, None
            if expected_features is None:
                expected_features = model_features 
            scaled_data = scaler_function(cus_data, scaler, expected_features)
            if scaled_data is None:
                logger.error("Failed to scale data")
                return None, None, None
            data_for_predict = scaled_data
        else:
            data_for_predict = cus_data

        # Check features for predict
        data_features = set(data_for_predict.columns)
        missing = set(model_features) - data_features
        if missing:
            logger.error(f"Missing {len(missing)} features required by model: {missing}")
            return None, None, None
        extra = data_features - set(model_features)
        if extra:
            logger.warning(f"Extra features in data: {extra}")
            data_for_predict = data_for_predict[model_features]

        logger.debug(f"Prediction input shape: {data_for_predict.shape}, columns: {list(data_for_predict.columns)}")

        predictions = model.predict(data_for_predict)
        if predictions is None or len(predictions) == 0:
            logger.error("Model predict returned None or empty array")
            return None, None, None
        
        probabilities = model.predict_proba(data_for_predict)[:, 1]
        if probabilities is None or len(probabilities) == 0:
            logger.error("Model predict_proba returned None or empty array")
            return None, None, None
        
        predictions_adjusted = (probabilities >= adjusted_threshold).astype(int)
        logger.info(f"Predictions made: predictions={predictions.tolist()}, "
                   f"probabilities={probabilities.tolist()}, adjusted={predictions_adjusted.tolist()}")
        return predictions, probabilities, predictions_adjusted

    except ValueError as ve:
        logger.error(f"ValueError in unified_prediction_function: {str(ve)}", exc_info=True)
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error in unified_prediction_function: {str(e)}", exc_info=True)
        return None, None, None

def predict_answers(cus_engineered_data: pd.DataFrame, model_path: str = None, scaler_path: str = None) -> Tuple[dict, int]:
    try:
        if not isinstance(cus_engineered_data, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(cus_engineered_data)}")
            raise ValueError(f"Input must be a pandas DataFrame, got {type(cus_engineered_data)}")
        if cus_engineered_data.empty:
            logger.error("Input data is empty")
            raise ValueError("Input data is empty")
        logger.debug(f"Input data shape: {cus_engineered_data.shape}, columns: {list(cus_engineered_data.columns)}")

        if model_path is None:
            model_path = Path("save_models/model.pkl")
        else:
            model_path = Path(model_path)
        
        if scaler_path is None:
            scaler_path = Path("save_models/scaler.pkl")
        else:
            scaler_path = Path(scaler_path)

        paths = get_artifact_paths(model_dir="save_models", model_path=str(model_path), scaler_path=str(scaler_path))
        
        scaler = load_scaler(scaler_path, paths.get('scaler_checksum'))
        model = load_model(model_path, paths.get('model_checksum'))

        if not hasattr(model, 'feature_names_in_'):
            logger.error("Model does not have feature_names_in_ attribute")
            raise ValueError("Model does not have feature_names_in_ attribute")
        expected_features = list(model.feature_names_in_)

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
            'adjust_prediction': int(predictions_adjusted[0]),
            'scaler_path': str(scaler_path),
            'model_path': str(model_path),
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