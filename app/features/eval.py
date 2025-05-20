from typing import Dict
import pickle
import os
from pathlib import Path
import pandas as pd
import logging
from app.data.data_transforming import data_transforming_fsk_v1
from app.data.data_engineering import data_engineering_fsk_v1
from app.predictions.predict import load_model, make_predictions
from sklearn.preprocessing import StandardScaler


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
    

def processAnswersFSK(answers: Dict) -> Dict:
   
    def validate_input(data: Dict, keys: list) -> None:
        missing = [key for key in keys if key not in data or not isinstance(data[key], list) or not data[key]]
        if missing:
            raise ValueError(f"Missing or invalid input fields: {missing}")

    def load_scaler(path: Path) -> StandardScaler:
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        with open(path, "rb") as file:
            scaler = pickle.load(file)
        if not isinstance(scaler, StandardScaler):
            raise TypeError("Invalid scaler object")
        return scaler

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

        scaler_path = Path(__file__).parent.parent / 'save_scaler' / 'scaler_rdf50_m1.0_fsk_f1.0.pkl'
        scaler = load_scaler(scaler_path)

        scaled_data = scaler.transform(engineered_data)
        scaled_df = pd.DataFrame(scaled_data, columns=engineered_data.columns)
        logger.debug(f"Scaled data shape: {scaled_df.shape}, columns: {list(scaled_df.columns)}")

        model = load_model("rdf50_m1.0_fsk_f1.0")
        predictions, probabilities, predictions_adjusted = make_predictions(model, scaled_df)

        results = {
            'default_probability': round(probabilities[0], 3),
            'model_prediction': int(predictions[0]),
            'adjust_prediction': int(predictions_adjusted[0])
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