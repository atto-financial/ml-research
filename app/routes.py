import logging
import os
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime
from typing import List, Tuple
from flask import request, jsonify, render_template
from joblib import dump, load
from flask import jsonify, request


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
        from sklearn.ensemble import RandomForestClassifier

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
                if not isinstance(model, RandomForestClassifier):
                    logger.error(f"Model is not a RandomForestClassifier: {type(model)}")
                    return pd.DataFrame()
                if not hasattr(model, 'feature_importances_'):
                    logger.error("Model lacks feature_importances_ attribute")
                    return pd.DataFrame()
                if len(X.columns) != len(model.feature_importances_):
                    logger.error(
                        f"Mismatch: X has {len(X.columns)} columns, but feature importances has {len(model.feature_importances_)}. Columns: {X.columns.tolist()}"
                    )
                    return pd.DataFrame()
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info(f"Feature Importance calculated successfully:\n{importance_df.to_string()}")
                return importance_df
            except ValueError as e:
                logger.error(f"Feature importance error: {str(e)}")
                return pd.DataFrame()
            except Exception as e:
                logger.error(f"Unexpected error in feature importance calculation: {str(e)}")
                return pd.DataFrame()

        def load_latest_model_metadata(model_dir: str) -> dict:
            metadata_path = os.path.join(model_dir, "model_metadata.json")
            try:
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                        roc_auc = metadata.get('cross_validated_roc_auc', metadata.get('roc_auc', 0.0))
                        if not isinstance(roc_auc, (int, float)) or np.isnan(roc_auc):
                            logger.warning(f"Invalid roc_auc in metadata: {roc_auc}. Setting to 0.0")
                            metadata['cross_validated_roc_auc'] = 0.0
                        else:
                            metadata['cross_validated_roc_auc'] = roc_auc
                       
                        if 'feature_importance' not in metadata:
                            logger.warning("No feature_importance in metadata. Setting to empty list.")
                            metadata['feature_importance'] = []
                        
                        for metric in ['cross_validated_accuracy', 'cross_validated_precision', 
                                     'cross_validated_recall', 'cross_validated_f1']:
                            if metric not in metadata:
                                logger.warning(f"No {metric} in metadata. Setting to 0.0")
                                metadata[metric] = 0.0
                        return metadata
                return {
                    "cross_validated_roc_auc": 0.0,
                    "model_path": "",
                    "checksum": "",
                    "feature_importance": [],
                    "cross_validated_accuracy": 0.0,
                    "cross_validated_precision": 0.0,
                    "cross_validated_recall": 0.0,
                    "cross_validated_f1": 0.0,
                }
            except Exception as e:
                logger.error(f"Failed to load model metadata: {str(e)}")
                return {
                    "cross_validated_roc_auc": 0.0,
                    "model_path": "",
                    "checksum": "",
                    "feature_importance": [],
                    "cross_validated_accuracy": 0.0,
                    "cross_validated_precision": 0.0,
                    "cross_validated_recall": 0.0,
                    "cross_validated_f1": 0.0,
                }

        def save_model_metadata(model_dir: str, roc_auc: float, model_path: str, checksum: str, 
                              feature_importance: pd.DataFrame, metrics: dict) -> None:
            metadata_path = os.path.join(model_dir, "model_metadata.json")
            try:
                
                feature_importance_dict = feature_importance.to_dict('records') if not feature_importance.empty else []
                if feature_importance.empty:
                    logger.warning("Feature importance is empty, saving empty list to metadata")
               
                required_metrics = ['cross_validated_accuracy', 'cross_validated_precision', 
                                  'cross_validated_recall', 'cross_validated_f1']
                metric_values = {}
                
                for metric in required_metrics:
                    value = metrics.get(metric, 0.0)
                    if not isinstance(value, (int, float)) or np.isnan(value):
                        logger.warning(f"Invalid {metric}: {value}. Setting to 0.0")
                        value = 0.0
                    metric_values[metric] = value
                metadata = {
                    "cross_validated_roc_auc": roc_auc,
                    "model_path": model_path,
                    "checksum": checksum,
                    "feature_importance": feature_importance_dict,
                    **metric_values
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                logger.info(f"Saved model metadata to {metadata_path} with feature importance and metrics")
            except Exception as e:
                logger.error(f"Failed to save model metadata: {str(e)}")
                raise

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scaler_dir = "save_scaler"
            model_dir = "save_models"
            output_dir = "output_data"

            CUSTOM_MODEL_NAME = ""  
            CUSTOM_SCALER_NAME = ""  
            # CUSTOM_IMPORTANCE_NAME = ""  

            model_filename = f"{CUSTOM_MODEL_NAME}.pkl" if CUSTOM_MODEL_NAME else f"custom_model_{timestamp}.pkl"
            scaler_filename = f"{CUSTOM_SCALER_NAME}.pkl" if CUSTOM_SCALER_NAME else f"custom_scaler_{timestamp}.pkl"
            # importance_filename = f"{CUSTOM_IMPORTANCE_NAME}.csv" if CUSTOM_IMPORTANCE_NAME else f"feature_importance_{timestamp}.csv"

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

            corr_dat = compute_correlations(scale_clean_engineer_dat)
            if corr_dat is None or corr_dat.empty:
                return jsonify({"error": "Failed to compute correlations"}), 500

            selected_features = select_top_features(corr_dat, n=10)
            if not selected_features:
                return jsonify({"error": "No features selected"}), 500

            selected_features = validate_features(selected_features, scale_clean_engineer_dat, "Selected features")
            if not selected_features:
                return jsonify({"error": "Invalid selected features"}), 500

            model, metrics = train_model(scale_clean_engineer_dat, selected_features)
            if model is None or metrics is None:
                return jsonify({"error": "Failed to train model"}), 500

            current_roc_auc = metrics.get('cross_validated_roc_auc', 0.0)
            if not isinstance(current_roc_auc, (int, float)) or np.isnan(current_roc_auc):
                logger.error(f"Invalid cross_validated_roc_auc: {current_roc_auc}")
                return jsonify({"error": "Invalid cross-validated ROC AUC"}), 500

            latest_metadata = load_latest_model_metadata(model_dir)
            latest_roc_auc = latest_metadata.get('cross_validated_roc_auc', latest_metadata.get('roc_auc', 0.0))

            scaler_path = ""
            scaler_checksum = ""
            model_path = ""
            model_checksum = ""
            # importance_path = ""

            logger.info(f"Comparing cross-validated ROC AUC: current={current_roc_auc:.3f}, latest={latest_roc_auc:.3f}")
            if current_roc_auc > latest_roc_auc:
                logger.info(f"New model ROC AUC {current_roc_auc:.3f} > latest {latest_roc_auc:.3f}, saving artifacts...")
                scaler_path, scaler_checksum = save_artifact(scaler, scaler_dir, scaler_filename, "scaler")
                model_path, model_checksum = save_artifact(model, model_dir, model_filename, "model")

                X = scale_clean_engineer_dat[metrics.get('best_features', selected_features)]
                importance_df = features_importance(model, X)
                if not importance_df.empty:
                #     os.makedirs(output_dir, exist_ok=True)
                #     importance_path = os.path.join(output_dir, importance_filename)
                #     importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
                #     logger.info(f"Saved feature importance to {importance_path}")
                # else:
                    logger.warning("Feature importance is empty")

                save_model_metadata(model_dir, current_roc_auc, model_path, model_checksum, importance_df, metrics)
            else:
                logger.info(f"New model ROC AUC {current_roc_auc:.3f} <= latest {latest_roc_auc:.3f}, skipping save.")

            scaler_instructions = (
                "To use the scaler for new data:\n"
                "1. Verify checksum: `from hashlib import sha256; with open('{}', 'rb') as f: assert sha256(f.read()).hexdigest() == '{}'`\n"
                "2. Load scaler: `from joblib import load; scaler = load('{}')`\n"
                "3. Transform data: `scaled_data = scaler.transform(new_data[features])`\n"
                "Ensure new_data contains the same features: {}".format(
                    scaler_path, scaler_checksum, scaler_path, metrics.get('best_features', selected_features)
                )
            ) if scaler_path else "No scaler saved (ROC AUC not improved)."

            response = {
                'model': str(model),
                'metrics': metrics,
                'scaler_path': scaler_path,
                'scaler_checksum': scaler_checksum,
                'model_path': model_path,
                'model_checksum': model_checksum,
                # 'feature_importance_path': importance_path,
                'final_features': metrics.get('best_features', selected_features),
                'scaler_instructions': scaler_instructions,
                'cross_validated_roc_auc': current_roc_auc,
                'latest_roc_auc': latest_roc_auc
            }
            return jsonify(response), 200

        except ValueError as ve:
            logger.error(f"ValueError: {str(ve)}")
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except Exception as e:
            logger.error(f"Internal Server Error: {str(e)}")
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
        
        
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
    