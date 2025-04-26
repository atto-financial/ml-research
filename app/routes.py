from flask import render_template, request, jsonify
from app.models.train_model import train_model
from app.models.evaluate_model import evaluate_model
from app.models.save_model import save_model
import pandas as pd
from app.config.settings import MODEL_PATH

def configure_routes(app):

    @app.route('/')
    def home():
        return render_template('train_model.html')

    @app.route('/eval', methods=['POST'])
    def userEval():
        from app.features.eval import processAnswersFeatureV2
        from app.models.eval import processAnswersModelV1
        request_body = request.get_json()
        app = request_body.get("application_label")
        
        if app == "cdd_f1.0_score" or app == "cdd_f1.0_criteria" or app == "rdf50_m1.0_cdd_f1.0":
            results = processAnswersModelV1(request_body.get("answers"))
            return jsonify(results), 200
    
        if app == "fkm_f1.0_score":
            feature_prediction = processAnswersFeatureV2(request_body.get("answers"))
            return jsonify({
                "feature_prediction": feature_prediction,
            }), 200   
             
        if app == "rdf50_m1.0_fkm_f1.0":
            # รอโมเดลใหม่
            # feature_prediction = processAnswersFeatureV2(request_body.answers)
            # model_predictions = processAnswersModelV2(request_body.answers)
            # use_feature = True
            # use_model = False
            # # application_label = "rdf50_m1.0_fkm_f1.0"
            # application_label = "fkm_f1.0_score"
            # return jsonify({
            #     "application_label": application_label,
            #     "feature_prediction": feature_prediction,
            # }), 200
             return jsonify({
                "msg": "coming soon"
            }), 200    
              
    @app.route('/train', methods=['POST'])
    def train():
        from app.data.load_data import load_data
        raw_dat = load_data()
        
        from app.data.preprocess import preprocess_data
        X_train, X_test, y_train, y_test, X, y = preprocess_data(raw_dat)

        model = train_model(X_train, y_train)

        results = evaluate_model(model, X_test, y_test)
        print("Evaluation results:", results)

        save_model(model, "app/models/model_x.pkl")
        return jsonify(results), 200

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()

        new_data = pd.DataFrame([data['cdd_vals']], columns=[f'cdd{i+1}' for i in range(len(data['cdd_vals']))])

        from app.predictions.predict import load_model
        model = load_model("model_s")

        from app.predictions.predict import make_predictions
        predictions, probabilities, predictions_adjusted = make_predictions(model, new_data)

        results = {
            'model_prediction': int(predictions[0]),
            'default_probability': float(f"{probabilities[0]:.3f}"),
            'adjust_prediction': int(predictions_adjusted)
        }

        return jsonify(results), 200
