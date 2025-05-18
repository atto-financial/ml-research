import logging
import pandas as pd
import os
from joblib import dump
from datetime import datetime
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
        #save_model(model, "app/models/mlrfth50_XXX.pkl")
        
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
        from app.models.rdf_auto import train_model,select_top_features

        @app.route('/rdftrain', methods=['POST'])
        def rdftrain():
            try:
                logging.debug(f"Received Content-Type: {request.content_type}")
                logging.debug(f"Request body: {request.get_data(as_text=True)}")
                if not request.is_json:
                    return jsonify({"error": "Bad Request: Request must contain valid JSON data"}), 400

                try:
                    data = request.get_json()
                    if data is None or not isinstance(data, dict):
                        return jsonify({"error": "Bad Request: JSON data is empty or invalid"}), 400
                except Exception as e:
                    return jsonify({"error": f"Bad Request: Invalid JSON data - {str(e)}"}), 400

                required_fields = ['model_filename', 'corr_filename', 'scaler_filename']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({"error": f"Bad Request: Missing fields: {', '.join(missing_fields)}"}), 400

                model_filename = data['model_filename']
                corr_filename = data['corr_filename']
                scaler_filename = data['scaler_filename']

                scaler_dir = "save_scaler"
                model_dir = "save_models"
                output_dir = "output_data"
                for directory in [scaler_dir, model_dir, output_dir]:
                    os.makedirs(directory, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                scaler_filename = f"{scaler_filename if scaler_filename else f'custom_scaler_{timestamp}'}.pkl"
                model_filename = f"{model_filename if model_filename else f'custom_model_{timestamp}'}.pkl"
                corr_filename = f"{corr_filename if corr_filename else f'correlations_with_ust_{timestamp}'}.csv"

                raw_dat = data_loading_fsk_v1()
                if raw_dat is None or raw_dat.empty:
                    return jsonify({"error": "Failed to load data: DataFrame is None or empty"}), 400

                clean_dat = data_cleaning_fsk_v1(raw_dat)
                if clean_dat is None or clean_dat.empty:
                    return jsonify({"error": "Failed to clean data: Missing required columns [fht1,fht2,fht3,fht4,fht5,fht6,fht7,fht8,set9,set10,kmsi1,kmsi2,kmsi3,kmsi4,kmsi5,kmsi6,kmsi7,kmsi8]"}), 400

                transform_dat = data_transforming_fsk_v1(clean_dat)
                if transform_dat is None or transform_dat.empty:
                    return jsonify({"error": "Failed to transform data: DataFrame is None or empty"}), 400

                engineer_dat = data_engineering_fsk_v1(transform_dat)
                if engineer_dat is None or engineer_dat.empty:
                    return jsonify({"error": "Failed to engineer data: DataFrame is None or empty"}), 400

                scale_clean_engineer_dat, scaler = data_preprocessing(engineer_dat)
                if scale_clean_engineer_dat is None or scale_clean_engineer_dat.empty:
                    return jsonify({"error": "Failed to preprocess data: DataFrame is None or empty"}), 400

                scaler_path = os.path.join(scaler_dir, scaler_filename)
                dump(scaler, scaler_path)

                corr_dat = compute_correlations(scale_clean_engineer_dat)
                if corr_dat is None or corr_dat.empty:
                    return jsonify({"error": "Failed to compute correlations: DataFrame is None or empty"}), 400

                corr_path = os.path.join(output_dir, corr_filename)
                corr_dat.to_csv(corr_path, index=True, encoding='utf-8-sig')
                
                selected_features = select_top_features(corr_dat, n=10)
                if not selected_features:
                    return jsonify({"error": "Failed to select features: No features selected"}), 400

                model, metrics = train_model(scale_clean_engineer_dat, selected_features)
                if model is None or metrics is None:
                    return jsonify({"error": "Failed to train model: Model or metrics is None"}), 400

                model_path = os.path.join(model_dir, model_filename)
                dump(model, model_path)

                combined_results = {
                    'model': str(model),
                    'metrics': metrics,
                    'scaler_path': scaler_path,
                    'corr_path': corr_path,
                    'model_path': model_path
                }

                return jsonify(combined_results), 200

            except ValueError as ve:
                return jsonify({"error": f"ValueError: {str(ve)}"}), 400
            except Exception as e:
                return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


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
    