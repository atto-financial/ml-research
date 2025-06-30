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
from app.predictions.ans_predictions import predict_answers
from app.utils_model import get_artifact_paths, load_and_verify_artifact, load


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
            raise ValueError(
                f"Key '{key}' must be a list, got {type(answers[key])}")


def set_answers_v1(data):
    answers = []
    for i in range(len(data)):
        value = int(data[i])
        answers.append(value)
    cus_ans_data = pd.DataFrame(
        [answers], columns=[f'set{i+9}' for i in range(len(answers))])
    from app.predictions.ans_predictions import load_model
    model = load_model("rdf50_m1.2_set_f1.0")
    from app.predictions.ans_predictions import make_predictions
    predictions, probabilities, predictions_adjusted = make_predictions(
        model, cus_ans_data)
    results = {
        'model_prediction': int(predictions[0]),
        'default_probability': float(f"{probabilities[0]:.3f}"),
        'adjust_prediction': int(predictions_adjusted)
    }
    return results


def set_answers_v2(answers):
    import pandas as pd

    fht = answers.get('fht')
    set = answers.get('set')
    kmsi = answers.get('kmsi')

    import pandas as pd

    cus_ans_data = pd.DataFrame(
        [set], columns=[f'set{i+9}' for i in range(len(set))])
    from app.predictions.ans_predictions import load_model
    model = load_model("rdf50_m1.2_set_f1.0")
    from app.predictions.ans_predictions import make_predictions
    predictions, probabilities, predictions_adjusted = make_predictions(
        model, cus_ans_data)
    results = {
        'default_probability': float(f"{probabilities[0]:.3f}"),
        'model_prediction': int(predictions[0]),
        'adjust_prediction': int(predictions_adjusted),
    }


def fsk_answers_v1(answers: Dict, model_path: str = None, scaler_path: str = None) -> Tuple[Dict, int]:
    try:
        required_keys = ['fht', 'set', 'kmsi']
        validate_input(answers, required_keys)
        fht, set_, kmsi = answers['fht'], answers['set'], answers['kmsi']
        fht = [int(x) for x in answers['fht']]
        set_ = [int(x) for x in answers['set']]
        kmsi = [int(x) for x in answers['kmsi']]
        print(fht)
        print(set_)
        print(kmsi)
        logger.debug(
            f"Validated input: fht={len(fht)}, set={len(set_)}, kmsi={len(kmsi)}")

        for i in range(len(kmsi)):
            if kmsi[i] in [1, 2]:
                kmsi[i] = 1
            elif kmsi[i] == 3:
                kmsi[i] = 2

        cus_ans = fht + set_ + kmsi
        columns = (
            [f'fht{i+1}' for i in range(len(fht))] +
            [f'set{i+1}' for i in range(len(set_))] +
            [f'kmsi{i+1}' for i in range(len(kmsi))]
        )
        cus_ans_data = pd.DataFrame([cus_ans], columns=columns)
        logger.debug(
            f"Input data shape: {cus_ans_data.shape}, columns: {list(cus_ans_data.columns)}")

        cus_transformed_data = data_transforming_fsk_v1(cus_ans_data)
        if not validate_data(cus_transformed_data, "Transformed data"):
            logger.error("Data transformation failed")
            raise ValueError("Data transformation failed")

        cus_engineered_data = data_engineering_fsk_v1(cus_transformed_data)
        if not validate_data(cus_engineered_data, "Engineered data"):
            logger.error("Feature engineering failed")
            raise ValueError("Feature engineering failed")
        logger.debug(
            f"Engineered data shape: {cus_engineered_data.shape}, columns: {list(cus_engineered_data.columns)}")

        non_numeric_cols = cus_engineered_data.select_dtypes(
            exclude=[np.number]).columns
        if non_numeric_cols.any():
            logger.error(
                f"Non-numeric columns in cus_engineered_data: {list(non_numeric_cols)}")
            raise ValueError(
                f"Non-numeric columns found: {list(non_numeric_cols)}")

        columns_to_drop = [
            col for col in cus_engineered_data.columns if col.endswith('_sum')]
        if columns_to_drop:
            logger.info(f"Dropping columns: {columns_to_drop}")
            cus_engineered_data = cus_engineered_data.drop(
                columns=columns_to_drop)
        logger.debug(
            f"Final engineered data shape: {cus_engineered_data.shape}, columns: {list(cus_engineered_data.columns)}")

        paths = get_artifact_paths(
            model_dir="save_models", model_path=model_path, scaler_path=scaler_path)
        model = load_and_verify_artifact(
            paths['model_path'], paths['model_checksum']) if paths['model_checksum'] else load(paths['model_path'])
        expected_features = paths.get('final_features', []) or (
            model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [])
        if isinstance(expected_features, (np.ndarray, pd.Series)):
            expected_features = expected_features.tolist()
        logger.debug(f"Expected features: {expected_features}")

        results, status = predict_answers(
            cus_engineered_data, model_path=model_path, scaler_path=scaler_path)
        return results, status

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}", exc_info=True)
        return {"error": f"ValueError: {str(ve)}"}, 400
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return {"error": f"Internal Server Error: {str(e)}"}, 500


def fsk_answers_v2(answers: Dict, model_path: str = None, scaler_path: str = None) -> Tuple[Dict, int]:
    try:
        required_keys = ['fht', 'set', 'kmsi']
        validate_input(answers, required_keys)
        fht, set_, kmsi = answers['fht'], answers['set'], answers['kmsi']
        fht = [int(x) for x in answers['fht']]
        set_ = [int(x) for x in answers['set']]
        kmsi = [int(x) for x in answers['kmsi']]
        print(fht)
        print(set_)
        print(kmsi)
        logger.debug(
            f"Validated input: fht={len(fht)}, set={len(set_)}, kmsi={len(kmsi)}")

        cus_ans = fht + set_ + kmsi
        columns = (
            [f'fht{i+1}' for i in range(len(fht))] +
            [f'set{i+1}' for i in range(len(set_))] +
            [f'kmsi{i+1}' for i in range(len(kmsi))]
        )
        cus_ans_data = pd.DataFrame([cus_ans], columns=columns)
        logger.debug(
            f"Input data shape: {cus_ans_data.shape}, columns: {list(cus_ans_data.columns)}")

        cus_transformed_data = data_transforming_fsk_v1(cus_ans_data)
        if not validate_data(cus_transformed_data, "Transformed data"):
            logger.error("Data transformation failed")
            raise ValueError("Data transformation failed")

        cus_engineered_data = data_engineering_fsk_v1(cus_transformed_data)
        if not validate_data(cus_engineered_data, "Engineered data"):
            logger.error("Feature engineering failed")
            raise ValueError("Feature engineering failed")
        logger.debug(
            f"Engineered data shape: {cus_engineered_data.shape}, columns: {list(cus_engineered_data.columns)}")

        non_numeric_cols = cus_engineered_data.select_dtypes(
            exclude=[np.number]).columns
        if non_numeric_cols.any():
            logger.error(
                f"Non-numeric columns in cus_engineered_data: {list(non_numeric_cols)}")
            raise ValueError(
                f"Non-numeric columns found: {list(non_numeric_cols)}")

        columns_to_drop = [
            col for col in cus_engineered_data.columns if col.endswith('_sum')]
        if columns_to_drop:
            logger.info(f"Dropping columns: {columns_to_drop}")
            cus_engineered_data = cus_engineered_data.drop(
                columns=columns_to_drop)
        logger.debug(
            f"Final engineered data shape: {cus_engineered_data.shape}, columns: {list(cus_engineered_data.columns)}")

        paths = get_artifact_paths(
            model_dir="save_models", model_path=model_path, scaler_path=scaler_path)
        model = load_and_verify_artifact(
            paths['model_path'], paths['model_checksum']) if paths['model_checksum'] else load(paths['model_path'])
        expected_features = paths.get('final_features', []) or (
            model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [])
        if isinstance(expected_features, (np.ndarray, pd.Series)):
            expected_features = expected_features.tolist()
        logger.debug(f"Expected features: {expected_features}")

        results, status = predict_answers(
            cus_engineered_data, model_path=model_path, scaler_path=scaler_path)
        return results, status

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}", exc_info=True)
        return {"error": f"ValueError: {str(ve)}"}, 400
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return {"error": f"Internal Server Error: {str(e)}"}, 500