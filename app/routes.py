import logging
import os
import pandas as pd
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from flask import request, jsonify, render_template
from joblib import dump, load

logger = logging.getLogger(__name__)


def configure_routes(app):
    @app.route('/')
    def home():
        return render_template('train_model.html')

    @app.route('/eval', methods=['POST'])
    def evaluate():
        from app.features.eval import processAnswersFeatureV2
        from app.models.eval import processAnswersModelV1
        from app.features.eval import processAnswersFSK

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

        if app_label == "rdf50_m1.0_fsk_f1.0":
          results = processAnswersFSK(answers)
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
        from app.models.rdf_auto import train_model, select_top_features
        
        def calculate_checksum(file_path: str) -> str:
        
            try:
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                return sha256_hash.hexdigest()
            except Exception as e:
                logger.error(f"Failed to calculate checksum for {file_path}: {str(e)}")
                raise
            
        def save_artifact(obj: object, directory: str, filename: str, obj_type: str) -> Tuple[str, str]:
            
            try:
                os.makedirs(directory, exist_ok=True)
                file_path = os.path.join(directory, filename)
                if isinstance(obj, pd.DataFrame):
                    obj.to_csv(file_path, index=False, encoding='utf-8-sig')
                    checksum = calculate_checksum(file_path)
                else:
                    dump(obj, file_path)  
                    checksum = calculate_checksum(file_path)
                logger.info(f"Saved {obj_type} to {file_path} with checksum: {checksum}")

                
                checksum_path = f"{file_path}.sha256"
                with open(checksum_path, 'w') as f:
                    f.write(checksum)
                logger.info(f"Saved checksum to {checksum_path}")

                return file_path, checksum
            except Exception as e:
                logger.error(f"Failed to save {obj_type}: {str(e)}")
                raise
            
        def load_and_verify_artifact(file_path: str, expected_checksum: str) -> object:
            
            try:
                current_checksum = calculate_checksum(file_path)
                if current_checksum != expected_checksum:
                    raise ValueError(f"Checksum mismatch for {file_path}: expected {expected_checksum}, got {current_checksum}")
                obj = load(file_path) 
                logger.info(f"Loaded and verified {file_path} with checksum: {current_checksum}")
                return obj
            except Exception as e:
                logger.error(f"Failed to load and verify {file_path}: {str(e)}")
                raise
            
        def validate_filename(filename: str, extension: str) -> str:
            
            sanitized = os.path.basename(filename).replace('..', '').replace('/', '').replace('\\', '')
            if not sanitized.endswith(extension):
                sanitized = f"{sanitized}.{extension}"
            return sanitized

        def validate_features(features: List[str], df: pd.DataFrame, name: str) -> List[str]:
            
            if not isinstance(features, list):
                logger.error(f"{name} is not a list: {type(features)}")
                return []
            missing = [f for f in features if f not in df.columns]
            if missing:
                logger.error(f"{name} not found in data: {missing}")
                return []
            return features

        def features_importance(model, X: pd.DataFrame) -> pd.DataFrame:
            
            try:
                if not hasattr(model, 'feature_importances_'):
                    raise ValueError("Model lacks feature_importances_ attribute")
                if len(X.columns) != len(model.feature_importances_):
                    raise ValueError(
                        f"Mismatch: X has {len(X.columns)} columns, but feature importances has {len(model.feature_importances_)}"
                    )
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info("Final Feature Importance:\n%s", importance_df.to_string())
                return importance_df
            except ValueError as e:
                logger.error(f"Feature importance error: {str(e)}")
                return pd.DataFrame()
      
        try:
            
            data = request.get_json()
            if not data or not isinstance(data, dict):
                return jsonify({"error": "Invalid or missing JSON"}), 400
    
            required_fields = ['model_filename', 'corr_filename', 'scaler_filename']
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400
    
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scaler_dir = "save_scaler"
            model_dir = "save_models"
            output_dir = "output_data"
    
            model_filename = validate_filename(data.get('model_filename', f'custom_model_{timestamp}'), 'pkl')
            corr_filename = validate_filename(data.get('corr_filename', f'correlations_with_ust_{timestamp}'), 'csv')
            scaler_filename = validate_filename(data.get('scaler_filename', f'custom_scaler_{timestamp}'), 'pkl')
            importance_filename = validate_filename(data.get('feature_importance_filename', f'feature_importance_{timestamp}'), 'csv')
    
            logger.info("Starting data pipeline...")
            raw_dat = data_loading_fsk_v1()
            if raw_dat is None or raw_dat.empty:
                return jsonify({"error": "Failed to load data"}), 500
    
            clean_dat = data_cleaning_fsk_v1(raw_dat)
            if clean_dat is None or clean_dat.empty:
                return jsonify({"error": "Failed to clean data"}), 500
    
            transform_dat = data_transforming_fsk_v1(clean_dat)
            if transform_dat is None or transform_dat.empty:
                return jsonify({"error": "Failed to transform data"}), 500
    
            engineer_dat = data_engineering_fsk_v1(transform_dat)
            if engineer_dat is None or engineer_dat.empty:
                return jsonify({"error": "Failed to engineer features"}), 500
    
            scale_clean_engineer_dat, scaler = data_preprocessing(engineer_dat)
            if scale_clean_engineer_dat is None or scaler is None:
                return jsonify({"error": "Failed to preprocess data"}), 500
    
            try:
                scaler.mean_  
            except AttributeError:
                return jsonify({"error": "Scaler is not fitted properly"}), 500
    
            scaler_path, scaler_checksum = save_artifact(scaler, scaler_dir, scaler_filename, "scaler")
    
            corr_dat = compute_correlations(scale_clean_engineer_dat)
            if corr_dat is None or corr_dat.empty:
                return jsonify({"error": "Failed to compute correlations"}), 500
            
            # corr_path, corr_checksum = save_artifact(corr_dat, output_dir, corr_filename, "correlations")
    
            selected_features = select_top_features(corr_dat, n=10)
            if not selected_features:
                return jsonify({"error": "No features selected"}), 500
    
            selected_features = validate_features(selected_features, scale_clean_engineer_dat, "Selected features")
            if not selected_features:
                return jsonify({"error": "Invalid selected features"}), 500
    
            model, metrics = train_model(scale_clean_engineer_dat, selected_features)
            if model is None or metrics is None:
                return jsonify({"error": "Failed to train model"}), 500
    
            model_path, model_checksum = save_artifact(model, model_dir, model_filename, "model")
    
            final_features = validate_features(metrics.get('best_features', selected_features), scale_clean_engineer_dat, "Final features")
            if not final_features:
                return jsonify({"error": "Invalid final features"}), 500
    
            X = scale_clean_engineer_dat[final_features]
            importance_df = features_importance(model, X)
            importance_path = ""
            importance_checksum = ""
            plot_path = ""
            if not importance_df.empty:
                importance_path, importance_checksum = save_artifact(importance_df, output_dir, importance_filename, "feature importance")
    
            scaler_instructions = (
                "To use the scaler for new data:\n"
                "1. Verify checksum: `from hashlib import sha256; with open('{}', 'rb') as f: assert sha256(f.read()).hexdigest() == '{}'`\n"
                "2. Load scaler: `from joblib import load; scaler = load('{}')`\n"
                "3. Transform data: `scaled_data = scaler.transform(new_data[features])`\n"
                "Ensure new_data contains the same features: {}".format(
                    scaler_path, scaler_checksum, scaler_path, final_features
                )
            )
    
            response = {
                'model': str(model),
                'metrics': metrics,
                'scaler_path': scaler_path,
                'scaler_checksum': scaler_checksum,
                'corr_path': corr_path,
                'corr_checksum': corr_checksum,
                'model_path': model_path,
                'model_checksum': model_checksum,
                'feature_importance_path': importance_path,
                'feature_importance_checksum': importance_checksum,
                'feature_importance_plot': plot_path,
                'final_features': final_features,
                'scaler_instructions': scaler_instructions
            }
            return jsonify(response), 200
    
        except ValueError as ve:
            logger.error(f"ValueError: {str(ve)}")
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except Exception as e:
            logger.error(f"Internal Server Error: {str(e)}")
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
    
    
    @app.route('/fskpredict', methods=['POST'])
    def fskpredict():
        from app.features.eval import processAnswersFSK
        try:
            data = request.get_json()
            if not isinstance(data, dict):
                raise ValueError("Invalid input: JSON object required")
            
            results = processAnswersFSK(data)
            
            if "error" in results:
                return jsonify(results), 400
            return jsonify(results), 200
        
        except ValueError as ve:
            logger.error(f"ValueError: {str(ve)}", exc_info=True)
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except Exception as e:
            logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    