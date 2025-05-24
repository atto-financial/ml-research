import json
import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from app.data.data_transforming import data_transforming_fsk_v1
from app.data.data_engineering import data_engineering_fsk_v1
from app.utils_model import validate_data
from app.predictions.predict import predict_fsk_answers


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



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

def validate_input(answers: Dict, required_keys: List[str]) -> None:
    """Validate that the input dictionary has required keys and list values."""
    missing_keys = [key for key in required_keys if key not in answers]
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
    for key in required_keys:
        if not isinstance(answers[key], list):
            raise ValueError(f"Key '{key}' must be a list, got {type(answers[key])}")

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
    """Process input answers, transform, engineer, and predict using specified or latest model."""
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
        if not validate_data(transformed_data, "Transformed data"):
            raise ValueError("Data transformation failed")

        engineered_data = data_engineering_fsk_v1(transformed_data)
        if not validate_data(engineered_data, "Engineered data"):
            raise ValueError("Feature engineering failed")
        logger.debug(f"Engineered data shape: {engineered_data.shape}, columns: {list(engineered_data.columns)}")

        # Drop sum columns
        columns_to_drop = [col for col in engineered_data.columns if col.endswith('_sum')]
        if columns_to_drop:
            logger.info(f"Dropping columns: {columns_to_drop}")
            engineered_data = engineered_data.drop(columns=columns_to_drop)
            engineered_data = engineered_data.astype(np.int64)

        # Predict using predict.py
        results, status_code = predict_fsk_answers(engineered_data, model_path, scaler_path)
        return results, status_code

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        return {"error": f"ValueError: {str(ve)}"}, 400
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}")
        return {"error": f"Internal Server Error: {str(e)}"}, 500