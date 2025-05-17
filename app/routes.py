import logging
import pandas as pd
from flask import render_template, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_routes(app):
    @app.route('/')
    def home():
        return render_template('train_model.html')

    @app.route('/eval', methods=['POST'])
    def evaluate():
        from app.features.eval import processAnswersFeatureV2
        from app.models.eval import processAnswersModelV1

        request_body = request.get_json()
        app_label = request_body.get("application_label")

        if not app_label:
            return jsonify({"msg": "application_label is required"}), 400

        answers = request_body.get("answers")

        if app_label in ["cdd_f1.0_score", "cdd_f1.0_criteria"]:
            results = processAnswersModelV1(answers)
            print(results)
            return jsonify(results), 200

        if app_label == "rdf50_m1.2_cdd_f1.0":
            results = processAnswersFeatureV2(answers)
            print(results)
            return jsonify(results), 200    

        return jsonify({
            "msg": f"Unsupported application_label: {app_label}"
        }), 400   
          
    @app.route('/train', methods=['POST'])
    def train():
        from app.data.data_loading import load_data_cdd
        from app.models.train_model import data_split
        from app.models.train_model import train_model
        from app.models.evaluate_model import test_set
        from app.models.evaluate_model import cross_validation
        from app.models.evaluate_model import features_importance
        
        raw_dat = load_data_cdd()
        
        X_train, X_test, y_train, y_test, X, y = data_split(raw_dat)
        
        model = train_model(X_train, y_train)
        
        test_results = test_set(model, X_test, y_test)
    
        CV_results = cross_validation(model, X_train, X_test, y_train, y_test, X, y)
       
        feature_results = features_importance(model, X, y)
        
        feature_results = feature_results.to_dict(orient='records') if feature_results is not None else []
        
        #from app.models.save_model import save_model
        #save_model(model, "app/models/mlrfth50_v1.2.pkl")
        
        combined_results = {
            'test_set': test_results,
            'cross_validation': CV_results,
            'feature_importance': feature_results
        }
        return jsonify(combined_results), 200
    
    @app.route('/rdftrain', methods=['POST'])
    def rdftrain():
        from app.data.data_loading import data_loading_fsk_v1
        from app.data.data_cleaning import data_cleaning_fsk_v1
        from app.data.data_transforming import data_transforming_fsk_v1
        from app.data.data_engineering import data_engineering_fsk_v1
        from app.data.data_preprocessing import data_preprocessing
        from app.models.correlation import compute_correlations
        from app.models.rdf_auto import train_model, select_top_features

        raw_dat = data_loading_fsk_v1()

        clean_dat = data_cleaning_fsk_v1(raw_dat)
            
        transform_dat = data_transforming_fsk_v1(clean_dat)

        engineer_dat = data_engineering_fsk_v1(transform_dat)

        clean_engineer_dat = data_preprocessing(engineer_dat)

        corr_dat = compute_correlations(clean_engineer_dat)

        selected_features = select_top_features(corr_dat, n=10)

        model, metrics = train_model(clean_engineer_dat, selected_features)

        combined_results = {
                'model': str(model),  
                'metrics': metrics
        }
        return jsonify(combined_results), 200


    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()

        new_data = pd.DataFrame([data['cdd_vals']], columns=[f'cdd{i+9}' for i in range(len(data['cdd_vals']))])

        from app.predictions.predict import load_model
        model = load_model("rdf50_m1.2_cdd_f1.0")

        from app.predictions.predict import make_predictions
        predictions, probabilities, predictions_adjusted = make_predictions(model, new_data)

        results = {
            'default_probability': float(f"{probabilities[0]:.3f}"),
            'model_prediction': int(predictions[0]),
            'adjust_prediction': int(predictions_adjusted)
        }
        return jsonify(results), 200
    