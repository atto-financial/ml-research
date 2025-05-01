import pandas as pd

def processAnswersModelV1(data):
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

def processAnswersModelV2(answers):
    # comming soon
    return 0
    
def processAnswersModelV3(answers):
    # comming soon
    return 0