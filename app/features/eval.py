import logging
import os
import pandas as pd
import numpy as np
import hashlib
from typing import Dict, List
from flask import jsonify
from sklearn.preprocessing import StandardScaler
from joblib import load
from datetime import datetime
import json



logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def processAnswersFeatureV1(answers):
    # cdd
    return 0


def processAnswersFeatureV2(answers):
    fht = answers.get('fht')
    cdd = answers.get('cdd')
    kmsi = answers.get('kmsi')
    
    # model eval
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

    # feature eval
    fhtScore = []
    for i in range(len(fht)):
        value = int(fht[i])
        if value == 1:
            fhtScore.append(3)
        elif value == 2:
            fhtScore.append(2)
        elif value == 3:
            fhtScore.append(1)
    cddScore = []
    for i in range(len(cdd)):
        value = int(cdd[i])
        if value == 1:
            cddScore.append(3)
        elif value == 2:
            cddScore.append(2)
        elif value == 3:
            cddScore.append(1)
    kmsiScore = []
    for i in range(len(kmsi)):
        value = int(kmsi[i])
        if i <= 5:
            if value == 1:
                kmsiScore.append(1)
            elif value == 2:
                kmsiScore.append(3)
        if i > 5 and i <= 7: 
            if value == 1:
                kmsiScore.append(3)
            elif value == 2:
                kmsiScore.append(1)
    sumFht = sum(fhtScore)
    sumKmsi = sum(kmsiScore)
    sumCdd = sum(cddScore)
    totalScore = sumFht + sumKmsi + sumCdd
    if totalScore >= 28:
        results['feature_prediction'] = 0
    else:
        results['feature_prediction'] = 1
        
    return results


def calculate_checksum(file_path: str) -> str:
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate checksum for {file_path}: {str(e)}")
        raise

def load_and_verify_artifact(file_path: str, expected_checksum: str) -> object:
    try:
        current_checksum = calculate_checksum(file_path)
        if current_checksum != expected_checksum:
            raise ValueError(f"Checksum mismatch for {file_path}: expected {expected_checksum}, got {current_checksum}")
        obj = load(file_path) 
        logger.info(f"Loaded and verified {file_path} with checksum: {current_checksum}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load and verify {file_path}: {str(e)}")
        raise

def validate_input(answers: Dict, required_keys: List[str]) -> None:
    missing_keys = [key for key in required_keys if key not in answers]
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
    for key in required_keys:
        if not isinstance(answers[key], list):
            raise ValueError(f"Key '{key}' must be a list, got {type(answers[key])}")

def get_artifact_paths(model_dir: str, scaler_dir: str) -> Dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    CUSTOM_MODEL_NAME = ""  
    CUSTOM_SCALER_NAME = "" 

    model_filename = f"{CUSTOM_MODEL_NAME}.pkl" if CUSTOM_MODEL_NAME else f"custom_model_{timestamp}.pkl"
    scaler_filename = f"{CUSTOM_SCALER_NAME}.pkl" if CUSTOM_SCALER_NAME else f"custom_scaler_{timestamp}.pkl"

    
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            model_path = metadata.get('model_path', os.path.join(model_dir, model_filename))
            model_checksum = metadata.get('checksum', '')
        else:
            model_path = os.path.join(model_dir, model_filename)
            model_checksum = ''
    except Exception as e:
        logger.error(f"Failed to load model metadata: {str(e)}")
        model_path = os.path.join(model_dir, model_filename)
        model_checksum = ''

    paths = {
        'scaler_path': os.path.join(scaler_dir, scaler_filename),
        'scaler_checksum': '',
        'model_path': model_path,
        'model_checksum': model_checksum
    }

    scaler_checksum_path = f"{paths['scaler_path']}.sha256"
    try:
        if os.path.exists(scaler_checksum_path):
            with open(scaler_checksum_path, 'r') as f:
                paths['scaler_checksum'] = f.read().strip()
    except Exception as e:
        logger.error(f"Failed to read scaler checksum: {str(e)}")

    os.makedirs(os.path.dirname(paths['scaler_path']), exist_ok=True)
    os.makedirs(os.path.dirname(paths['model_path']), exist_ok=True)

    for key, path in paths.items():
        if key.endswith('_path') and not os.path.exists(os.path.dirname(path)):
            raise FileNotFoundError(f"Directory for {key} does not exist: {os.path.dirname(path)}")

    return paths


def processAnswersFSK(answers: Dict) -> Dict:
    from app.data.data_transforming import data_transforming_fsk_v1
    from app.data.data_engineering import data_engineering_fsk_v1
    from app.models.rdf_auto import make_predictions

    try:
        validate_input(answers, ['fht', 'set', 'kmsi'])
        fht, set_, kmsi = answers['fht'], answers['set'], answers['kmsi']
        logger.debug(f"Validated input: fht={len(fht)}, set={len(set_)}, kmsi={len(kmsi)}")

        all_values = fht + set_ + kmsi
        columns = (
            [f'fht{i+1}' for i in range(len(fht))] +
            [f'set{i+1}' for i in range(len(set_))] +
            [f'kmsi{i+1}' for i in range(len(kmsi))]
        )
        new_data = pd.DataFrame([all_values], columns=columns)
        logger.debug(f"Input data shape: {new_data.shape}, columns: {list(new_data.columns)}")

        transformed_data = data_transforming_fsk_v1(new_data)
        if transformed_data is None or transformed_data.empty:
            raise ValueError("Data transformation failed")

        engineered_data = data_engineering_fsk_v1(transformed_data)
        if engineered_data is None or engineered_data.empty:
            raise ValueError("Feature engineering failed")
        logger.debug(f"Engineered data shape: {engineered_data.shape}, columns: {list(engineered_data.columns)}")

        columns_to_drop = [col for col in engineered_data.columns if col.endswith('_sum')]
        if columns_to_drop:
            engineered_data = engineered_data.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns: {columns_to_drop}")
            engineered_data = engineered_data.astype(np.int64)

        paths = get_artifact_paths(model_dir="save_models", scaler_dir="save_scaler")
        logger.debug(f"Scaler path: {paths['scaler_path']}")
        logger.debug(f"Model path: {paths['model_path']}")


        if paths['scaler_checksum']:
            scaler = load_and_verify_artifact(paths['scaler_path'], paths['scaler_checksum'])
        else:
            scaler = load(paths['scaler_path'])
            logger.info(f"No checksum for scaler, loaded without verification: {paths['scaler_path']}")

        if not isinstance(scaler, StandardScaler):
            raise TypeError("Invalid scaler object")

        
        if paths['model_checksum']:
            model = load_and_verify_artifact(paths['model_path'], paths['model_checksum'])
        else:
            model = load(paths['model_path'])
            logger.info(f"No checksum for model, loaded without verification: {paths['model_path']}")

        scaled_data = scaler.transform(engineered_data)
        scaled_df = pd.DataFrame(scaled_data, columns=engineered_data.columns)
        logger.debug(f"Scaled data shape: {scaled_df.shape}, columns: {list(scaled_df.columns)}")

        predictions, probabilities, predictions_adjusted = make_predictions(model, scaled_df)

        results = {
            'default_probability': round(float(probabilities[0]), 3),
            'model_prediction': int(predictions[0]),
            'adjust_prediction': int(predictions_adjusted[0]),
            'scaler_path': paths['scaler_path'],
            'scaler_checksum': paths['scaler_checksum'],
            'model_path': paths['model_path'],
            'model_checksum': paths['model_checksum']
        }
        logger.info(f"Prediction results: {results}")
        return results

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}", exc_info=True)
        return {"error": f"ValueError: {str(ve)}"}
    except FileNotFoundError as fnf:
        logger.error(f"FileNotFoundError: {str(fnf)}", exc_info=True)
        return {"error": f"FileNotFoundError: {str(fnf)}"}
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return {"error": f"Internal Server Error: {str(e)}"}