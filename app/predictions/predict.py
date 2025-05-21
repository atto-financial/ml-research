import pandas as pd
import joblib
from pathlib import Path

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(str(model_path))

def make_predictions(model, new_data, threshold=0.75):
    import logging
    try:
        print('engineered_data', new_data.columns.tolist())
        # Ensure new_data has only the features the model was trained on
        model_features = set(model.feature_names_in_)
        data_features = set(new_data.columns)
        unseen = list(data_features - model_features)
        missing = list(model_features - data_features)
        
        if unseen:
            logging.warning(f"Unseen features at prediction time: {unseen}")
            # Drop unseen features
            new_data = new_data.drop(columns=unseen, errors='ignore')
        
        if missing:
            logging.error(f"Missing features required by model: {missing}")
            return None, None, None
        
        # Ensure column order matches training
        new_data = new_data[model.feature_names_in_]
        
        predictions = model.predict(new_data)
        probabilities = model.predict_proba(new_data)[:, 1]
        predictions_adjusted = (probabilities >= threshold).astype(int)
        return predictions, probabilities, predictions_adjusted
    except Exception as e:
        logging.error(f"Error in make_predictions: {e}", exc_info=True)
        return None, None, None