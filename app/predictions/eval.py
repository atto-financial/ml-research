import json
import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from app.data.data_transforming import data_transforming_fsk_v1
from app.data.data_engineering import data_engineering_fsk_v1
from app.predictions.predict import make_predictions
from app.utils_model import load_and_verify_artifact, get_artifact_paths, validate_features, load


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def validate_input(answers: Dict, required_keys: List[str]) -> None:
    """Validate that the input dictionary has required keys and list values."""
    missing_keys = [key for key in required_keys if key not in answers]
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
    for key in required_keys:
        if not isinstance(answers[key], list):
            raise ValueError(f"Key '{key}' must be a list, got {type(answers[key])}")


def get_artifact_paths(model_dir: str) -> Dict:
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    default_paths = {
        'scaler_path': '',
        'scaler_checksum': '',
        'model_path': '',
        'model_checksum': '',
        'final_features': []
    }
    
    try:
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")
            return default_paths
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        paths = {
            'scaler_path': metadata.get('scaler_path', ''),
            'scaler_checksum': metadata.get('scaler_checksum', ''),
            'model_path': metadata.get('model_path', ''),
            'model_checksum': metadata.get('model_checksum', ''),
            'final_features': metadata.get('final_features', [])
        }
        
        # Verify paths
        for key in ['scaler_path', 'model_path']:
            if paths[key] and not os.path.exists(paths[key]):
                logger.error(f"{key} does not exist: {paths[key]}")
                paths[key] = ''
                paths[key.replace('_path', '_checksum')] = ''
        
        return paths
    except Exception as e:
        logger.error(f"Failed to load artifact paths: {str(e)}")
        return default_paths


def set_answers_v1(data):
    answers = []
    for i in range(len(data)):
        value = int(data[i])
        answers.append(value)
    new_data = pd.DataFrame([answers], columns=[f'cdd{i+9}' for i in range(len(answers))])
    from app.predictions.predict import load_model
    model = load_model("rdf50_m1.2_cdd_f1.0")
    from app.predictions.predict import make_predictions
    predictions, probabilities, predictions_adjusted = make_predictions(model, new_data)
    results = {
        'model_prediction': int(predictions[0]),
        'default_probability': float(f"{probabilities[0]:.3f}"),
        'adjust_prediction': int(predictions_adjusted)
    }
    return results


def set_answers_v2(answers):
    import pandas as pd
    
    fht = answers.get('fht')
    cdd = answers.get('cdd')
    kmsi = answers.get('kmsi')
    
    import pandas as pd
    
    new_data = pd.DataFrame([cdd], columns=[f'cdd{i+9}' for i in range(len(cdd))])
    from app.predictions.predict import load_model
    model = load_model("rdf50_m1.2_cdd_f1.0")
    from app.predictions.predict import make_predictions
    predictions, probabilities, predictions_adjusted = make_predictions(model, new_data)
    results = {
        'default_probability': float(f"{probabilities[0]:.3f}"),
        'model_prediction': int(predictions[0]),
        'adjust_prediction': int(predictions_adjusted),
    }


def fsk_answers_v1(answers: Dict, model_path: str = None, scaler_path: str = None) -> Tuple[Dict, int]:
    """Process input answers and make predictions using specified or best model."""
    try:
        # Validate input
        required_keys = ['fht', 'set', 'kmsi']
        validate_input(answers, required_keys)
        fht, set_, kmsi = answers['fht'], answers['set'], answers['kmsi']
        logger.debug(f"Validated input: fht={len(fht)}, set={len(set_)}, kmsi={len(kmsi)}")

        # Create DataFrame
        all_values = fht + set_ + kmsi
        columns = (
            [f'fht{i+1}' for i in range(len(fht))] +
            [f'set{i+1}' for i in range(len(set_))] +
            [f'kmsi{i+1}' for i in range(len(kmsi))]
        )
        new_data = pd.DataFrame([all_values], columns=columns)
        logger.debug(f"Input data shape: {new_data.shape}, columns: {list(new_data.columns)}")

        # Data transformation and engineering
        transformed_data = data_transforming_fsk_v1(new_data)
        if transformed_data is None or transformed_data.empty:
            raise ValueError("Data transformation failed")

        engineered_data = data_engineering_fsk_v1(transformed_data)
        if engineered_data is None or engineered_data.empty:
            raise ValueError("Feature engineering failed")
        logger.debug(f"Engineered data shape: {engineered_data.shape}, columns: {list(engineered_data.columns)}")

        # Drop sum columns
        columns_to_drop = [col for col in engineered_data.columns if col.endswith('_sum')]
        if columns_to_drop:
            logger.info(f"Dropping columns: {columns_to_drop}")
            engineered_data = engineered_data.drop(columns=columns_to_drop)
            engineered_data = engineered_data.astype(np.int64)

        # Load artifacts
        paths = get_artifact_paths(model_dir="save_models", model_path=model_path, scaler_path=scaler_path)
        logger.debug(f"Scaler path: {paths['scaler_path']}, Model path: {paths['model_path']}")

        # Load scaler
        try:
            if not paths['scaler_path']:
                raise FileNotFoundError("Scaler path not found")
            scaler = load_and_verify_artifact(paths['scaler_path'], paths['scaler_checksum']) if paths['scaler_checksum'] else load(paths['scaler_path'])
            if not isinstance(scaler, StandardScaler):
                raise TypeError(f"Invalid scaler object: {type(scaler)}")
            logger.info(f"Scaler loaded: {paths['scaler_path']}")
        except FileNotFoundError as fnf:
            logger.error(f"Scaler file not found: {str(fnf)}")
            return {"error": f"Scaler file not found: {str(fnf)}"}, 404

        # Load model
        try:
            if not paths['model_path']:
                raise FileNotFoundError("Model path not found")
            model = load_and_verify_artifact(paths['model_path'], paths['model_checksum']) if paths['model_checksum'] else load(paths['model_path'])
            logger.info(f"Model loaded: {paths['model_path']}")
        except FileNotFoundError as fnf:
            logger.error(f"Model file not found: {str(fnf)}")
            return {"error": f"Model file not found: {str(fnf)}"}, 404

        # Validate features against model
        expected_features = paths.get('final_features', [])
        if not expected_features:
            logger.warning("No final_features found in metadata. Using all columns.")
        else:
            valid_features = validate_features(expected_features, engineered_data, "Model features")
            if not valid_features:
                missing = set(expected_features) - set(engineered_data.columns)
                raise ValueError(f"Missing required features: {missing}")
            engineered_data = engineered_data[valid_features]
            logger.debug(f"Selected features for prediction: {valid_features}")

        # Scale data
        try:
            scaled_data = scaler.transform(engineered_data)
            scaled_df = pd.DataFrame(scaled_data, columns=engineered_data.columns)
            logger.debug(f"Scaled data shape: {scaled_df.shape}, columns: {list(scaled_df.columns)}")
        except Exception as e:
            logger.error(f"Failed to scale data: {str(e)}")
            return {"error": f"Failed to scale data: {str(e)}"}, 400

        # Make predictions
        try:
            predictions, probabilities, predictions_adjusted = make_predictions(model, scaled_df)
        except Exception as e:
            logger.error(f"Failed to make predictions: {str(e)}")
            return {"error": f"Failed to make predictions: {str(e)}"}, 400

        results = {
            'default_probability': round(float(probabilities[0]), 3),
            'model_prediction': int(predictions[0]),
            'adjust_prediction': int(predictions_adjusted[0]),
            'scaler_path': paths['scaler_path'],
            'scaler_checksum': paths['scaler_checksum'],
            'model_path': paths['model_path'],
            'model_checksum': paths['model_checksum'],
            'used_features': list(engineered_data.columns)
        }
        logger.info(f"Prediction results: {results}")
        return results, 200

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}", exc_info=True)
        return {"error": f"ValueError: {str(ve)}"}, 400
    except TypeError as te:
        logger.error(f"TypeError: {str(te)}", exc_info=True)
        return {"error": f"TypeError: {str(te)}"}, 400
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return {"error": f"Internal Server Error: {str(e)}"}, 500