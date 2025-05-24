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
from app.utils_model import validate_data, validate_features, features_importance, load_latest_model_metadata, save_all_artifacts, ensure_paths


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
        """Train a RandomForestClassifier and save artifacts if performance improves."""
        try:
            # Initialize paths with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            paths = {
                'scaler': ("save_scalers", f"custom_scaler_{timestamp}.pkl"),
                'model': ("save_models", f"custom_model_{timestamp}.pkl"),
                'output': ("output_data", f"feature_importance_{timestamp}.csv")
            }
    
            # Ensure directories exist
            paths_ok, error_msg = ensure_paths(paths)
            if not paths_ok:
                return jsonify({"error": f"Failed to create directories: {error_msg}"}), 500
    
            # Run data pipeline
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
                if not validate_data(raw_dat, f"Data after {step}"):
                    return jsonify({"error": error_msg}), 500
    
            # Preprocess data
            scale_clean_engineer_dat, scaler = data_preprocessing(raw_dat)
            if not validate_data(scale_clean_engineer_dat, "Preprocessed data") or scaler is None or not hasattr(scaler, 'mean_'):
                return jsonify({"error": "Failed to preprocess data or scaler not fitted"}), 500
    
            # Select features
            corr_dat = compute_correlations(scale_clean_engineer_dat)
            if not validate_data(corr_dat, "Correlation data"):
                return jsonify({"error": "Failed to compute correlations"}), 500
    
            selected_features = validate_features(select_top_features(corr_dat, n=10), scale_clean_engineer_dat, "Selected features")
            if not selected_features:
                return jsonify({"error": "No features selected"}), 500
            logger.info(f"Initial selected features: {selected_features}")
    
            # Train model
            model, metrics = train_model(scale_clean_engineer_dat, selected_features)
            if model is None or metrics is None:
                return jsonify({"error": "Failed to train model"}), 500
    
            # Get final features from RFE
            final_features = metrics.get('best_features', selected_features)
            if not validate_features(final_features, scale_clean_engineer_dat, "Final features"):
                return jsonify({"error": f"Invalid final features: {final_features}"}), 500
            logger.info(f"Final features after RFE: {final_features}")
    
            # Calculate feature importance
            importance_df = features_importance(model, scale_clean_engineer_dat[final_features])
            if importance_df.empty:
                logger.warning("Feature importance calculation returned empty DataFrame")
    
            # Extract and validate metrics
            cv_roc_auc = float(metrics.get('cross_validated_roc_auc', 0.0))
            if np.isnan(cv_roc_auc) or cv_roc_auc < 0.0:
                logger.warning(f"Invalid cv_roc_auc: {cv_roc_auc}. Setting to 0.0")
                cv_roc_auc = 0.0
    
            metrics_summary = {
                'cv_roc_auc': cv_roc_auc,
                'cv_accuracy': float(metrics.get('cross_validated_accuracy', 0.0)),
                'cv_precision': float(metrics.get('cross_validated_precision', 0.0)),
                'cv_recall': float(metrics.get('cross_validated_recall', 0.0)),
                'cv_f1': float(metrics.get('cross_validated_f1', 0.0))
            }
            logger.info(f"Metrics: {metrics_summary}")
    
            # Compare with latest model
            latest_metadata = load_latest_model_metadata(paths['model'][0])
            latest_cv_roc_auc = latest_metadata.get('cv_roc_auc', 0.0)
            metadata_exists = bool(latest_metadata.get('model_path'))
            logger.info(f"Metadata exists: {metadata_exists}, latest_cv_roc_auc: {latest_cv_roc_auc:.3f}")
    
            # Save artifacts if first run or model improves
            artifact_info = {
                'scaler_path': '', 'scaler_checksum': '',
                'model_path': '', 'model_checksum': '',
                'importance_path': '', 'importance_checksum': ''
            }
            if not metadata_exists or cv_roc_auc >= latest_cv_roc_auc:
                logger.info(f"Saving artifacts: first run={not metadata_exists}, cv_roc_auc {cv_roc_auc:.3f} >= latest {latest_cv_roc_auc:.3f}")
                artifact_info = save_all_artifacts(scaler, model, importance_df, final_features, metrics, paths, timestamp)
                if artifact_info is None:
                    return jsonify({"error": "Failed to save artifacts"}), 500
    
            # Prepare scaler instructions
            scaler_instructions = (
                f"To use the scaler:\n"
                f"1. Verify checksum: `from hashlib import sha256; with open('{artifact_info['scaler_path']}', 'rb') as f: assert sha256(f.read()).hexdigest() == '{artifact_info['scaler_checksum']}'`\n"
                f"2. Load scaler: `from joblib import load; scaler = load('{artifact_info['scaler_path']}')`\n"
                f"3. Transform data: `scaled_data = scaler.transform(new_data[final_features])`\n"
                f"Ensure new_data contains: {final_features}"
            ) if artifact_info['scaler_path'] else "No scaler saved (cv_roc_auc not improved)."
    
            # Prepare response
            response = {
                'model': str(model),
                'metrics': {
                    **metrics_summary,
                    'classification_report': metrics.get('classification_report', {})
                },
                'final_features': final_features,
                'artifacts': artifact_info,
                'scaler_instructions': scaler_instructions,
                'latest_cv_roc_auc': latest_cv_roc_auc
            }
            return jsonify(response), 200
    
        except ValueError as ve:
            logger.error(f"ValueError in rdftrain: {str(ve)}")
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except (OSError, PermissionError) as e:
            logger.error(f"System Error in rdftrain: {str(e)}")
            return jsonify({"error": f"System Error: {str(e)}"}), 500
        except Exception as e:
            logger.error(f"Internal Server Error in rdftrain: {str(e)}")
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500