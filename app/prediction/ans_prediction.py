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
from app.utils_model import get_artifact_paths, load_and_verify_artifact, load
from app.prediction.utils_prediction import screen_fsk_answers, predict_answers

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def validate_input(answers: Dict, required_keys: List[str]) -> None:
    
    missing_keys = [key for key in required_keys if key not in answers]
    if missing_keys:
        logger.warning(f"Missing keys (proceeding dynamically): {', '.join(missing_keys)}")
    for key in answers:
        if key in required_keys and not isinstance(answers[key], list):
            logger.warning(f"Key '{key}' is not a list (proceeding): {type(answers[key])}")

def fsk_answers_v2(answers: Dict, model_path: str = None, scaler_path: str = None) -> Tuple[Dict, int]:
    try:
        required_keys = ['fht', 'set', 'kmsi']
        validate_input(answers, required_keys)
        fht = [int(x) for x in answers.get('fht', [])]
        set_ = [int(x) for x in answers.get('set', [])]
        kmsi = [int(x) for x in answers.get('kmsi', [])]
        print(fht)
        print(set_)
        print(kmsi)
        logger.debug(
            f"Validated input: fht={len(fht)}, set={len(set_)}, kmsi={len(kmsi)}")

        screening_passed, screening_msg = screen_fsk_answers(answers)

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
        
        results['screening_passed'] = screening_passed
        results['screening_msg'] = screening_msg
        results['checksum_valid'] = True

        return results, status

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}", exc_info=True)
        return {"error": f"ValueError: {str(ve)}"}, 400
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return {"error": f"Internal Server Error: {str(e)}"}, 500
    
def fk_answers_v1(answers: Dict, model_path: str = None, scaler_path: str = None) -> Tuple[Dict, int]:
    try:
        required_keys = ['fht','kmsi']
        validate_input(answers, required_keys)
        fht = [int(x) for x in answers.get('fht', [])]
        kmsi = [int(x) for x in answers.get('kmsi', [])]
        print(fht)
        print(kmsi)
        logger.debug(
            f"Validated input: fht={len(fht)}, kmsi={len(kmsi)}")

        screening_passed, screening_msg = screen_fsk_answers(answers)

        cus_ans = fht + kmsi
        columns = (
            [f'fht{i+1}' for i in range(len(fht))] +
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
        
        results['screening_passed'] = screening_passed
        results['screening_msg'] = screening_msg
        results['checksum_valid'] = True

        return results, status

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}", exc_info=True)
        return {"error": f"ValueError: {str(ve)}"}, 400
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return {"error": f"Internal Server Error: {str(e)}"}, 500