import pandas as pd
import joblib
from app.config.settings import MODEL_PATH

def load_model():
    return joblib.load(MODEL_PATH)

def make_predictions(model, new_data):
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]
    return predictions, probabilities
