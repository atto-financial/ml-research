import pandas as pd
import joblib
from pathlib import Path

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(str(model_path))

def make_predictions(model, new_data, threshold=0.75):
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]
    predictions_adjusted = (probabilities >= threshold).astype(int)
    return predictions, probabilities, predictions_adjusted
