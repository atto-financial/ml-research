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
    
def processAnswersFeatureV3(answers):
    # comming soon
    return 0