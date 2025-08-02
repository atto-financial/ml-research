import json
from flask import request, jsonify, render_template
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from app.data.data_loading import data_loading_fsk_v1
from app.data.data_cleaning import data_cleaning_fsk_v1
from app.data.data_transforming import data_transforming_fsk_v1
from app.data.data_engineering import data_engineering_fsk_v1
from app.data.data_preprocessing import data_preprocessing
from app.models.correlation import compute_correlations
from app.models.lucis import train_model, select_top_features, ModelConfig
from app.utils_model import ensure_paths, load_latest_model_metadata, save_all_artifacts, features_importance, validate_data, validate_features, setup_paths
from app.prediction.ans_prediction import fk_answers_v1
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from app.utils.feature import extract_feature_answers
import sklearn
import joblib

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def run_data_pipeline() -> Optional[pd.DataFrame]:
    pipeline = [
        ("load", data_loading_fsk_v1, "Failed to load data"),
        ("clean", data_cleaning_fsk_v1, "Failed to clean data"),
        ("transform", data_transforming_fsk_v1, "Failed to transform data"),
        ("engineer", data_engineering_fsk_v1, "Failed to engineer features")
    ]
    raw_dat = None
    for step, func, error_msg in pipeline:
        try:
            logger.debug(f"Running pipeline step: {step}")
            raw_dat = func() if step == "load" else func(raw_dat)
            if not validate_data(raw_dat, f"Data after {step}"):
                logger.error(error_msg)
                return None
        except Exception as e:
            logger.error(f"{error_msg}: {str(e)}")
            return None
    return raw_dat

def preprocess_and_select_features(data: pd.DataFrame, config: ModelConfig) -> Tuple[Optional[pd.DataFrame], Optional[StandardScaler], List[str]]:
    try:
        scale_clean_engineer_dat, scaler = data_preprocessing(data)
        if not validate_data(scale_clean_engineer_dat, "Preprocessed data") or scaler is None:
            logger.error("Preprocessing failed or scaler is None")
            return None, None, []
        if not isinstance(scaler, StandardScaler) or not hasattr(scaler, 'mean_'):
            logger.error("Scaler is not a fitted StandardScaler")
            return None, None, []

        corr_dat = compute_correlations(scale_clean_engineer_dat)
        if not validate_data(corr_dat, "Correlation data"):
            logger.error("Failed to compute correlations")
            return None, None, []

        selected_features = validate_features(select_top_features(
            corr_dat, config), scale_clean_engineer_dat, "Selected features")
        if not selected_features:
            logger.error("No features selected")
            return None, None, []

        return scale_clean_engineer_dat, scaler, selected_features
    except Exception as e:
        logger.error(f"Error in preprocessing and feature selection: {str(e)}")
        return None, None, []

def train_and_evaluate_model(data: pd.DataFrame, features: List[str], config: ModelConfig) -> Tuple[Optional[RandomForestClassifier], Optional[Dict], List[str]]:
    try:
        model, metrics = train_model(data, features, config)
        if model is None or metrics is None:
            logger.error("Failed to train model")
            return None, None, []

        final_features = metrics.get('best_features', features)
        if not validate_features(final_features, data, "Final features"):
            logger.error(f"Invalid final features: {final_features}")
            return None, None, []

        return model, metrics, final_features
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return None, None, []

def get_scaler_instructions(artifact_info: Dict, final_features: List[str], package_versions: Dict, decision_threshold: float) -> str:
    if not artifact_info['scaler_path']:
        return "No scaler saved (model performance not improved)."

    return (
        f"To use the scaler for unsecured loan approval:\n"
        f"1. Install compatible package versions:\n"
        f"   pip install scikit-learn=={package_versions['sklearn']} joblib=={package_versions['joblib']}\n"
        f"2. Load scaler: `from joblib import load; scaler = load('{artifact_info['scaler_path']}')`\n"
        f"3. Verify checksum: `from app.utils_model import calculate_checksum; assert calculate_checksum('{artifact_info['scaler_path']}') == '{artifact_info['scaler_checksum']}'`\n"
        f"4. Transform data: `scaled_data = scaler.transform(cus_ans_data[{final_features}])`\n"
        f"5. Predict with model: `proba = model.predict_proba(scaled_data)[:, 1]; approved = proba >= {decision_threshold}`\n"
        f"Ensure cus_ans_data contains columns: {final_features}\n"
        f"Note: Scaler was trained with scikit-learn {package_versions['sklearn']} and joblib {package_versions['joblib']}. "
        f"Use decision threshold {decision_threshold} for approval."
    )

def configure_routes(app):
    # @app.route('/') 
    # def home():
    #     return render_template('train_model.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            if not data:
                logger.error("No JSON data provided in request")
                return jsonify({"error": "No JSON data provided"}), 400
            application_label = data.get('application_label', '')
            answers = data.get('answers', {})
            model_path = data.get('model_path', None)
            scaler_path = data.get('scaler_path', None)
            metadata_path = data.get('metadata_path', None)
            
            logger.info(f"raw_answers : {answers}")
            answers_order = [
                {"group": "fht", "version": "1"},
                {"group": "kmsi", "version": "1"}
            ]
            answers = extract_feature_answers(answers, answers_order)
            logger.info(f"extracted_answers : {answers}")

            if not isinstance(answers, dict):
                logger.error(f"Invalid answers format: {type(answers)}")
                return jsonify({"error": "Answers must be a dictionary"}), 400
            if not answers:
                logger.error("Answers dictionary is empty")
                return jsonify({"error": "Answers dictionary is empty"}), 400

            if application_label == "rdf50_v3.0_fk_v1.0":
                results, status = fk_answers_v1(
                    answers, metadata_path=metadata_path, model_path=model_path, scaler_path=scaler_path
                )
                if status == 200:
                    logger.info(f"Prediction successful for application_label: {application_label}")
                return jsonify(results), status
            else:
                logger.error(f"Unsupported application_label: {application_label}")
                return jsonify({"error": f"Unsupported application_label: {application_label}"}), 400
        except Exception as e:
            logger.error(f"Error in /predict: {str(e)}", exc_info=True)
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

    @app.route('/lucis', methods=['POST'])
    def lucis():
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            config = ModelConfig() 

            paths = setup_paths(timestamp)
            paths_ok, error_msg = ensure_paths(paths)
            if not paths_ok:
                logger.error(f"Failed to create directories: {error_msg}")
                return jsonify({"error": f"Failed to create directories: {error_msg}"}), 500

            logger.info(
                {"message": "Starting random forest pipeline", "timestamp": timestamp})

            raw_dat = run_data_pipeline()
            if raw_dat is None:
                logger.error("Data pipeline failed")
                return jsonify({"error": "Data pipeline failed"}), 500

            # required_features = ['xxx', 'xxx', 'xxx']
            # missing_features = [f for f in required_features if f not in raw_dat.columns]
            # if missing_features:
            #     logger.error(f"Missing required features for loan approval: {missing_features}")
            #     return jsonify({"error": f"Missing required features: {missing_features}"}), 400

            scale_clean_engineer_dat, scaler, selected_features = preprocess_and_select_features(
                raw_dat, config)
            if scale_clean_engineer_dat is None or scaler is None or not selected_features:
                logger.error("Preprocessing or feature selection failed")
                return jsonify({"error": "Preprocessing or feature selection failed"}), 500
            logger.info({"message": "Selected features",
                        "features": selected_features})

            model, metrics, final_features = train_and_evaluate_model(
                scale_clean_engineer_dat, selected_features, config)
            if model is None or metrics is None:
                logger.error("Failed to train random forest model")
                return jsonify({"error": "Failed to train random forest model"}), 500
            logger.info({"message": "Final features Model selected",
                        "features": final_features})

            importance_df = features_importance(
                model, scale_clean_engineer_dat[final_features])
            if importance_df.empty:
                logger.warning(
                    {"message": "Feature importance calculation returned empty DataFrame"})

            metrics_summary = {
                'cv_roc_auc': float(metrics.get('cross_validated_roc_auc', 0.0)),
                'cv_accuracy': float(metrics.get('cross_validated_accuracy', 0.0)),
                'cv_precision': float(metrics.get('cross_validated_precision', 0.0)),
                'cv_recall': float(metrics.get('cross_validated_recall', 0.0)),
                'cv_f1': float(metrics.get('cross_validated_f1', 0.0)),
                'cv_score_mean': float(metrics.get('cv_score_mean', 0.0)),
                'cv_score_std': float(metrics.get('cv_score_std', 0.0)),
                'test_roc_auc': float(metrics.get('test_roc_auc', 0.0)),
                'test_accuracy': float(metrics.get('test_accuracy', 0.0)),
                'test_precision': float(metrics.get('test_precision', 0.0)),
                'test_recall': float(metrics.get('test_recall', 0.0)),
                'test_f1': float(metrics.get('test_f1', 0.0)),
                'accuracy_ci_lower': float(metrics.get('accuracy_ci_lower', np.nan)),
                'accuracy_ci_upper': float(metrics.get('accuracy_ci_upper', np.nan)),
                'precision_ci_lower': float(metrics.get('precision_ci_lower', np.nan)),
                'precision_ci_upper': float(metrics.get('precision_ci_upper', np.nan)),
                'recall_ci_lower': float(metrics.get('recall_ci_lower', np.nan)),
                'recall_ci_upper': float(metrics.get('recall_ci_upper', np.nan)),
                'f1_ci_lower': float(metrics.get('f1_ci_lower', np.nan)),
                'f1_ci_upper': float(metrics.get('f1_ci_upper', np.nan)),
                'roc_auc_ci_lower': float(metrics.get('roc_auc_ci_lower', np.nan)),
                'roc_auc_ci_upper': float(metrics.get('roc_auc_ci_upper', np.nan)),
                'overfitting_gap': float(abs(metrics.get('cross_validated_accuracy', 0.0) - metrics.get('test_accuracy', 0.0)))
            }
            for metric, value in metrics_summary.items():
                if np.isnan(value) or value < 0.0:
                    logger.warning(
                        {"message": f"Invalid {metric}", "value": value})
                    metrics_summary[metric] = 0.0
            logger.info({"message": "Training metrics",
                        "metrics": metrics_summary})

            valid_scoring_metrics = ['accuracy',
                                     'precision', 'recall', 'f1', 'roc_auc']
            if config.scoring not in valid_scoring_metrics:
                logger.error(
                    f"Invalid scoring metric: {config.scoring}. Must be one of {valid_scoring_metrics}")
                return jsonify({"error": f"Invalid scoring metric: {config.scoring}"}), 400

            scoring_key = f'cv_{config.scoring}'
            if scoring_key not in metrics_summary:
                logger.error(
                    f"Metric {scoring_key} not found in metrics_summary")
                return jsonify({"error": f"Metric {scoring_key} not found in metrics"}), 500

            latest_metadata = load_latest_model_metadata(paths['model'][0])
            latest_cv_scoring = latest_metadata.get(scoring_key, 0.0)
            metadata_exists = bool(latest_metadata.get('model_path'))
            logger.info({
                "message": "Model comparison",
                "metadata_exists": metadata_exists,
                "trained_cv_scoring": metrics_summary[scoring_key],
                "latest_cv_scoring": latest_cv_scoring
            })

            improved = not metadata_exists or (metrics_summary[scoring_key] > latest_cv_scoring and metrics_summary['cv_score_std'] < 0.10)
            if improved:
                logger.info({
                    "message": "Saving artifacts",
                    "first_run": not metadata_exists,
                    f"cv_{config.scoring}": metrics_summary[scoring_key],
                    "latest_cv_scoring": latest_cv_scoring
                })
                artifact_info = save_all_artifacts(
                    scaler, model, importance_df, final_features, metrics_summary, paths, timestamp
                )
                if artifact_info is None:
                    logger.error("Failed to save artifacts")
                    return jsonify({"error": "Failed to save artifacts"}), 500
            else:
                artifact_info = {
                    'scaler_path': latest_metadata.get('scaler_path', ''),
                    'scaler_checksum': latest_metadata.get('scaler_checksum', ''),
                    'model_path': latest_metadata.get('model_path', ''),
                    'model_checksum': latest_metadata.get('model_checksum', ''),
                    'metadata_path': latest_metadata.get('metadata_path', ''),
                    'metadata_checksum': latest_metadata.get('metadata_checksum', '')
                }

            latest_metadata = load_latest_model_metadata(paths['model'][0])

            package_versions = latest_metadata.get('package_versions', {
                'sklearn': sklearn.__version__,
                'joblib': joblib.__version__
            })
            decision_threshold = latest_metadata.get('decision_threshold', config.decision_threshold)
            final_features_list = [item['feature'] for item in latest_metadata.get('final_features', [])]
            scaler_instructions = get_scaler_instructions(
                artifact_info, final_features_list, package_versions, decision_threshold)

            final_features_dict = latest_metadata.get('final_features', [])
    
            response = {
                '0.model': str(model) if improved else f"Using existing model from {latest_metadata.get('model_path', 'unknown')}",
                '1.latest_cv_scoring': latest_metadata.get(scoring_key, 0.0),
                '2.metrics': {
                    'cv_roc_auc': latest_metadata['cv_roc_auc'],
                    'cv_accuracy': latest_metadata['cv_accuracy'],
                    'cv_precision': latest_metadata['cv_precision'],
                    'cv_recall': latest_metadata['cv_recall'],
                    'cv_f1': latest_metadata['cv_f1'],
                    'test_roc_auc': latest_metadata['test_roc_auc'],
                    'test_accuracy': latest_metadata['test_accuracy'],
                    'test_precision': latest_metadata['test_precision'],
                    'test_recall': latest_metadata['test_recall'],
                    'test_f1': latest_metadata['test_f1'],
                    'cv_score_std': latest_metadata['cv_score_std'],
                    'overfitting_gap': latest_metadata['overfitting_gap'],
                },
                '3.confidence_intervals': {
                    'accuracy': {
                        'lower': latest_metadata['accuracy_ci_lower'],
                        'upper': latest_metadata['accuracy_ci_upper']
                    },
                    'precision': {
                        'lower': latest_metadata['precision_ci_lower'],
                        'upper': latest_metadata['precision_ci_upper']
                    },
                    'recall': {
                        'lower': latest_metadata['recall_ci_lower'],
                        'upper': latest_metadata['recall_ci_upper']
                    },
                    'f1': {
                        'lower': latest_metadata['f1_ci_lower'],
                        'upper': latest_metadata['f1_ci_upper']
                    },
                    'roc_auc': {
                        'lower': latest_metadata['roc_auc_ci_lower'],
                        'upper': latest_metadata['roc_auc_ci_upper']
                    }
                },
                '4.final_features': final_features_dict,
                '5.artifacts': artifact_info,
                '6.package_versions': package_versions,
                '7.scaler_instructions': scaler_instructions,
                '8.model_config': {
                    'scoring': latest_metadata['scoring'],
                    'n_estimators': latest_metadata['n_estimators'],
                    'max_depth': latest_metadata['max_depth'],
                    'min_samples_split': latest_metadata['min_samples_split'],
                    'n_features_to_select': latest_metadata['n_features_to_select'],
                    'n_folds': latest_metadata['n_folds'],
                    'random_state': latest_metadata['random_state'],
                    'oversampling_method': latest_metadata['oversampling_method'],
                    'categorical_features': latest_metadata['categorical_features'],
                    'use_undersampling': latest_metadata['use_undersampling'],
                    'max_vif': latest_metadata['max_vif'],
                    'n_bootstraps': latest_metadata['n_bootstraps'],
                    'alpha': latest_metadata['alpha'],
                    'test_size': latest_metadata['test_size'],
                    'memory_threshold': latest_metadata['memory_threshold'],
                    'cost_weight': latest_metadata['cost_weight'],
                    'decision_threshold': latest_metadata['decision_threshold']
                },
                '9.timestamp': latest_metadata.get('timestamp', timestamp)
            }

            logger.info(
                {"message": "Training completed successfully", "response": response})
            return jsonify(response), 200

        except ValueError as ve:
            logger.error({"message": "ValueError in lucis", "error": str(ve)})
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except (OSError, PermissionError) as e:
            logger.error({"message": "System Error in lucis", "error": str(e)})
            return jsonify({"error": f"System Error: {str(e)}"}), 500
        except Exception as e:
            logger.error(
                {"message": "Internal Server Error in lucis", "error": str(e)}, exc_info=True)
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

    @app.route('/test/run_data_pipeline', methods=['POST'])
    def data_load():
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config = ModelConfig() 
            paths = setup_paths(timestamp)
            paths_ok, error_msg = ensure_paths(paths)
            if not paths_ok:
                logger.error(f"Failed to create directories: {error_msg}")
                return jsonify({"error": f"Failed to create directories: {error_msg}"}), 500

            logger.info(
                {"message": "Starting random forest pipeline", "timestamp": timestamp})

            raw_dat = run_data_pipeline()
            json_compatible_data = raw_dat.to_dict(orient='records')  # 'records' format is common for JSON
            return jsonify(json_compatible_data)
            
        except ValueError as ve:
            logger.error({"message": "ValueError in lucis", "error": str(ve)})
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except (OSError, PermissionError) as e:
            logger.error({"message": "System Error in lucis", "error": str(e)})
            return jsonify({"error": f"System Error: {str(e)}"}), 500
        except Exception as e:
            logger.error(
                {"message": "Internal Server Error in lucis", "error": str(e)}, exc_info=True)
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500