import pandas as pd
import joblib

def load_model(modelName):
    return joblib.load("app/models/" + modelName + ".pkl")

def make_predictions(model, new_data):
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]

    threshold = 0.75 
    predictions_adjusted = (probabilities >= threshold).astype(int)
   
    return predictions, probabilities, predictions_adjusted
