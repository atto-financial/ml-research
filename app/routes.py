from flask import request, jsonify, render_template
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from app.data.data_loading import data_loading_fsk_v1
from app.data.data_cleaning import data_cleaning_fsk_v1
from app.data.data_transforming import data_transforming_fsk_v1
from app.data.data_engineering import data_engineering_fsk_v1
from app.data.data_preprocessing import data_preprocessing
from app.models.correlation import compute_correlations
from app.models.rdf_auto import train_model, select_top_features
from app.utils_model import save_artifact, validate_features, features_importance, load_latest_model_metadata, save_model_metadata

logger = logging.getLogger(__name__)

def configure_routes(app):
    @app.route('/')
    def home():
        return render_template('train_model.html')

    @app.route('/eval', methods=['POST'])
    def evaluate():
        from app.predictions.eval import set_answers_v2
        from app.predictions.eval import set_answers_v1
        from app.predictions.eval import fsk_answers_v1

        request_body = request.get_json()
        app_label = request_body.get("application_label")

        if not app_label:
            return jsonify({"msg": "application_label is required"}), 400

        answers = request_body.get("answers")

        if app_label in ["cdd_f1.0_score", "cdd_f1.0_criteria"]:
            results = set_answers_v1(answers)
            print(results)
            return jsonify(results), 200

        if app_label == "rdf50_m1.2_cdd_f1.0":
            results = set_answers_v2(answers)
            print(results)
            return jsonify(results), 200    

        if app_label == "rdf50_m1.0_fsk_f1.0":
          results = fsk_answers_v1(answers)
          print(results)
          return jsonify(results), 200    

        return jsonify({
            "msg": f"Unsupported application_label: {app_label}"
        }), 400   
        
        
    @app.route('/fskpredict', methods=['POST'])
    def fskpredict():
        from app.predictions.eval import fsk_answers_v1
        try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400

                application_label = data.get('application_label', '')
                answers = data.get('answers', {})
                model_path = data.get('model_path', None)
                scaler_path = data.get('scaler_path', None)

                result, status = fsk_answers_v1(answers, model_path=model_path, scaler_path=scaler_path)
                result['application_label'] = application_label
                return jsonify(result), status
        except Exception as e:
            logger.error(f"Error in /eval: {str(e)}")
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    
    @app.route('/rdftrain', methods=['POST'])
    def rdftrain():
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            paths = {
                'scaler': ("save_scaler", f"custom_scaler_{timestamp}.pkl"),
                'model': ("save_models", f"custom_model_{timestamp}.pkl"),
                'output': ("output_data", f"feature_importance_{timestamp}.csv")
            }

            # Data pipeline
            logger.info("Starting data pipeline...")
            pipeline = [
                ("load", data_loading_fsk_v1, "Failed to load data"),
                ("clean", data_cleaning_fsk_v1, "Failed to clean data"),
                ("transform", data_transforming_fsk_v1, "Failed to transform data"),
                ("engineer", data_engineering_fsk_v1, "Failed to engineer features")
            ]
            raw_dat = None
            for step, func, error_msg in pipeline:
                raw_dat = func() if step == "load" else func(raw_dat)
                if raw_dat is None or raw_dat.empty:
                    return jsonify({"error": error_msg}), 500

            # Preprocessing
            scale_clean_engineer_dat, scaler = data_preprocessing(raw_dat)
            if scale_clean_engineer_dat is None or scaler is None or not hasattr(scaler, 'mean_'):
                return jsonify({"error": "Failed to preprocess data or scaler not fitted"}), 500

            # Feature selection
            corr_dat = compute_correlations(scale_clean_engineer_dat)
            if corr_dat is None or corr_dat.empty:
                return jsonify({"error": "Failed to compute correlations"}), 500

            selected_features = validate_features(select_top_features(corr_dat, n=10), scale_clean_engineer_dat, "Selected features")
            if not selected_features:
                return jsonify({"error": "No features selected"}), 500

            # Model training
            model, metrics = train_model(scale_clean_engineer_dat, selected_features)
            if model is None or metrics is None:
                return jsonify({"error": "Failed to train model"}), 500

            # Use cross_validated_roc_auc
            current_cv_roc_auc = metrics.get('cross_validated_roc_auc', 0.0)
            logger.info(f"Current cv_roc_auc from metrics: {current_cv_roc_auc}")
            if not isinstance(current_cv_roc_auc, (int, float)) or np.isnan(current_cv_roc_auc):
                logger.warning(f"Invalid cross_validated_roc_auc: {current_cv_roc_auc}. Setting to 0.0")
                current_cv_roc_auc = 0.0

            # Extract other metrics
            cv_accuracy = metrics.get('cross_validated_accuracy', 0.0)
            cv_precision = metrics.get('cross_validated_precision', 0.0)
            cv_recall = metrics.get('cross_validated_recall', 0.0)
            cv_f1 = metrics.get('cross_validated_f1', 0.0)
            logger.info(f"Metrics: accuracy={cv_accuracy}, precision={cv_precision}, recall={cv_recall}, f1={cv_f1}")

            # Compare and save model
            latest_metadata = load_latest_model_metadata(paths['model'][0])
            latest_cv_roc_auc = latest_metadata.get('cv_roc_auc', 0.0)
            metadata_exists = os.path.exists(os.path.join(paths['model'][0], "model_metadata.json"))
            logger.info(f"Metadata exists: {metadata_exists}, latest_cv_roc_auc: {latest_cv_roc_auc}")

            scaler_path, scaler_checksum, model_path, model_checksum, importance_path = "", "", "", "", ""

            # Save artifacts for first run or if model improves
            if not metadata_exists or current_cv_roc_auc >= latest_cv_roc_auc:
                logger.info(f"Saving artifacts: first run={not metadata_exists}, cv_roc_auc {current_cv_roc_auc:.3f} >= latest {latest_cv_roc_auc:.3f}")

                # Ensure directories exist
                for path_type in paths:
                    try:
                        os.makedirs(paths[path_type][0], exist_ok=True)
                        logger.info(f"Created directory: {paths[path_type][0]}")
                    except Exception as e:
                        logger.error(f"Failed to create directory {paths[path_type][0]}: {str(e)}")
                        return jsonify({"error": f"Failed to create directory {paths[path_type][0]}: {str(e)}"}), 500

                # Save scaler
                try:
                    scaler_path, scaler_checksum = save_artifact(scaler, *paths['scaler'], "scaler")
                    logger.info(f"Scaler saved at {scaler_path} with checksum {scaler_checksum}")
                except Exception as e:
                    logger.error(f"Failed to save scaler: {str(e)}")
                    return jsonify({"error": f"Failed to save scaler: {str(e)}"}), 500

                # Save model
                try:
                    model_path, model_checksum = save_artifact(model, *paths['model'], "model")
                    logger.info(f"Model saved at {model_path} with checksum {model_checksum}")
                except Exception as e:
                    logger.error(f"Failed to save model: {str(e)}")
                    return jsonify({"error": f"Failed to save model: {str(e)}"}), 500

                # Save feature importance
                best_features = metrics.get('best_features', selected_features)
                if not validate_features(best_features, scale_clean_engineer_dat, "Best features"):
                    return jsonify({"error": f"Best features not found: {best_features}"}), 500

                importance_df = features_importance(model, scale_clean_engineer_dat[best_features])
                if not importance_df.empty:
                    try:
                        importance_path, importance_checksum = save_artifact(importance_df, *paths['output'], "feature_importance")
                        logger.info(f"Feature importance saved at {importance_path} with checksum {importance_checksum}")
                    except Exception as e:
                        logger.error(f"Failed to save feature importance: {str(e)}")
                        importance_path = ""
                        importance_checksum = ""
                else:
                    logger.warning("Feature importance is empty")
                    importance_path = ""
                    importance_checksum = ""

                # Save model metadata
                try:
                    save_model_metadata(
                        paths['model'][0], 
                        current_cv_roc_auc, 
                        model_path, 
                        model_checksum, 
                        scaler_path, 
                        scaler_checksum, 
                        importance_df, 
                        {
                            'cv_accuracy': cv_accuracy,
                            'cv_precision': cv_precision,
                            'cv_recall': cv_recall,
                            'cv_f1': cv_f1,
                            'final_features': best_features
                        },
                        filename=f"model_metadata_{timestamp}.json"  # เปลี่ยนชื่อไฟล์
                    )
                    logger.info("Metadata saved successfully")
                except Exception as e:
                    logger.error(f"Failed to save metadata: {str(e)}")
                    return jsonify({"error": f"Failed to save metadata: {str(e)}"}), 500
                else:
                    logger.info(f"Skipping save: cv_roc_auc {current_cv_roc_auc:.3f} < latest {latest_cv_roc_auc:.3f}")

            # Scaler instructions
            scaler_instructions = (
                f"To use the scaler:\n"
                f"1. Verify checksum: `from hashlib import sha256; with open('{scaler_path}', 'rb') as f: assert sha256(f.read()).hexdigest() == '{scaler_checksum}'`\n"
                f"2. Load scaler: `from joblib import load; scaler = load('{scaler_path}')`\n"
                f"3. Transform data: `scaled_data = scaler.transform(new_data[features])`\n"
                f"Ensure new_data contains: {metrics.get('best_features', selected_features)}"
            ) if scaler_path else "No scaler saved (cv_roc_auc not improved or invalid)."

            return jsonify({
                'model': str(model),
                'metrics': metrics,
                'scaler_path': scaler_path,
                'scaler_checksum': scaler_checksum,
                'model_path': model_path,
                'model_checksum': model_checksum,
                'feature_importance_path': importance_path,
                'feature_importance': importance_df.to_dict('records') if not importance_df.empty else [],
                'final_features': metrics.get('best_features', selected_features),
                'scaler_instructions': scaler_instructions,
                'cv_roc_auc': current_cv_roc_auc,
                'latest_cv_roc_auc': latest_cv_roc_auc,
                'cv_accuracy': cv_accuracy,
                'cv_precision': cv_precision,
                'cv_recall': cv_recall,
                'cv_f1': cv_f1
            }), 200

        except ValueError as ve:
            logger.error(f"ValueError in rdftrain: {str(ve)}")
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except Exception as e:
            logger.error(f"Internal Server Error in rdftrain: {str(e)}")
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500