import logging
import pandas as pd
import numpy as np
import os
from joblib import dump
from datetime import datetime
from flask import render_template, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
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
                logger.error(f"Feature importance error: {str(e)}", exc_info=True)
                return pd.DataFrame()
    
        def validate_features(features: list, df: pd.DataFrame, name: str) -> list:
            
            if not isinstance(features, list):
                logger.error(f"{name} is not a list: {type(features)}")
                return selected_features
            missing = [f for f in features if f not in df.columns]
            if missing:
                logger.error(f"{name} not found in data: {missing}")
                return selected_features
            return features
    
        def save_artifact(obj, path: str, obj_type: str) -> None:
            
            try:
                if isinstance(obj, pd.DataFrame):
                    obj.to_csv(path, index=False, encoding='utf-8-sig')
                else:
                    dump(obj, path)
                logger.info(f"Saved {obj_type} to {path}")
            except Exception as e:
                logger.error(f"Failed to save {obj_type}: {str(e)}", exc_info=True)
    
        try:
            
            data = request.get_json()
            if not data or not isinstance(data, dict):
                return jsonify({"error": "Invalid or missing JSON"}), 400
    
            required_fields = ['model_filename', 'corr_filename', 'scaler_filename']
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

            scaler_dir = "save_scaler"
            model_dir = "save_models"
            output_dir = "output_data"
            for dir_path in [scaler_dir, model_dir, output_dir]:
                os.makedirs(dir_path, exist_ok=True)
    
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{data.get('model_filename', f'custom_model_{timestamp}')}.pkl"
            corr_filename = f"{data.get('corr_filename', f'correlations_with_ust_{timestamp}')}.csv"
            scaler_filename = f"{data.get('scaler_filename', f'custom_scaler_{timestamp}')}.pkl"
            importance_filename = f"{data.get('feature_importance_filename', f'feature_importance_{timestamp}')}.csv"
    
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
    
            scaler_path = os.path.join(scaler_dir, scaler_filename)
            # save_artifact(scaler, scaler_path, "scaler")
    
            corr_dat = compute_correlations(scale_clean_engineer_dat)
            if corr_dat is None or corr_dat.empty:
                return jsonify({"error": "Failed to compute correlations"}), 500
    
            corr_path = os.path.join(output_dir, corr_filename)
            # save_artifact(corr_dat, corr_path, "correlations")
            
            selected_features = select_top_features(corr_dat, n=10)
            if not selected_features or not isinstance(selected_features, list):
                return jsonify({"error": f"Invalid selected features: {selected_features}"}), 500
    
            missing_features = [f for f in selected_features if f not in scale_clean_engineer_dat.columns]
            if missing_features:
                return jsonify({"error": f"Selected features not found: {missing_features}"}), 500
            
            model, metrics = train_model(scale_clean_engineer_dat, selected_features)
            if model is None or metrics is None:
                return jsonify({"error": "Failed to train model"}), 500
    
            model_path = os.path.join(model_dir, model_filename)
            save_artifact(model, model_path, "model")
    
            final_features = validate_features(
                metrics.get('best_features', selected_features),
                scale_clean_engineer_dat,
                "Final features"
            )
            logger.debug(f"Using final_features: {final_features}")
    
            X = scale_clean_engineer_dat[final_features]
            logger.debug(f"X shape: {X.shape}, features: {list(X.columns)}")
            importance_df = features_importance(model, X)
    
            importance_path = None
            if not importance_df.empty:
                importance_path = os.path.join(output_dir, importance_filename)
                save_artifact(importance_df, importance_path, "feature importance")
            else:
                logger.warning("Feature importance calculation failed")
    
            return jsonify({
                'model': str(model),
                'metrics': metrics,
                'scaler_path': scaler_path,
                'corr_path': corr_path,
                'model_path': model_path,
                'feature_importance_path': importance_path,
                'final_features': final_features
            }), 200
    
        except ValueError as ve:
            logger.error(f"ValueError: {str(ve)}", exc_info=True)
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except Exception as e:
            logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
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
    