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
from app.predictions.ans_transformimg import set_answers_v1, set_answers_v2, fsk_answers_v1, fsk_answers_v2
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
            raw_dat = func() if step == "load" else func(raw_dat)
            if not validate_data(raw_dat, f"Data after {step}"):
                logger.error(error_msg)
                return None
        except Exception as e:
            logger.error(f"{error_msg}: {str(e)}")
            return None
    return raw_dat


def preprocess_and_select_features(data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[StandardScaler], List[str]]:
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
            corr_dat, n=10), scale_clean_engineer_dat, "Selected features")
        if not selected_features:
            logger.error("No features selected")
            return None, None, []

        return scale_clean_engineer_dat, scaler, selected_features
    except Exception as e:
        logger.error(f"Error in preprocessing and feature selection: {str(e)}")
        return None, None, []


def train_and_evaluate_model(data: pd.DataFrame, features: List[str]) -> Tuple[Optional[RandomForestClassifier], Optional[Dict], List[str]]:
    try:
        model, metrics = train_model(data, features)
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


def get_scaler_instructions(artifact_info: Dict, final_features: List[str], package_versions: Dict) -> str:

    if not artifact_info['scaler_path']:
        return "No scaler saved (cv_roc_auc not improved)."

    return (
        f"To use the scaler:\n"
        f"1. Install compatible package versions:\n"
        f"   pip install scikit-learn=={package_versions['sklearn']} joblib=={package_versions['joblib']}\n"
        f"2. Load scaler: `from joblib import load; scaler = load('{artifact_info['scaler_path']}')`\n"
        f"3. Verify checksum: `from your_module import calculate_checksum; assert calculate_checksum('{artifact_info['scaler_path']}') == '{artifact_info['scaler_checksum']}'`\n"
        f"4. Transform data: `scaled_data = scaler.transform(cus_ans_data[{final_features}])`\n"
        f"Ensure cus_ans_data contains columns: {final_features}\n"
        f"Note: Scaler was trained with scikit-learn {package_versions['sklearn']} and joblib {package_versions['joblib']}."
    )


def configure_routes(app):
    @app.route('/')
    def home():
        return render_template('train_model.html')

    @app.route('/eval', methods=['POST'])
    def evaluate():
        print("Received request on /eval")
        try:
            request_body = request.get_json()
            if not request_body:
                logger.error("No JSON data provided in request")
                return jsonify({"error": "No JSON data provided"}), 400

            app_label = request_body.get("application_label")
            if not app_label:
                logger.error(
                    f"Missing application_label in request: {request_body}")
                return jsonify({"msg": "application_label is required"}), 400

            answers = request_body.get("answers")
            if not answers or not isinstance(answers, dict):
                logger.error(
                    f"Invalid or missing answers in request: {request_body}")
                return jsonify({"msg": "answers is required and must be a dictionary"}), 400

            model_path = request_body.get("model_path")
            scaler_path = request_body.get("scaler_path")

            if app_label in ["set_f1.0_score", "set_f1.0_criteria"]:
                results = set_answers_v1(answers)
                logger.info(f"Processed {app_label}: {results}")
                return jsonify(results), 200

            elif app_label == "rdf50_m1.2_set_f1.0":
                results = set_answers_v2(answers)
                logger.info(f"Processed {app_label}: {results}")
                return jsonify(results), 200

            elif app_label == "rdf50_m1.0_fsk_f1.0":
                results, status = fsk_answers_v1(
                    answers, model_path=model_path, scaler_path=scaler_path)
                results['application_label'] = app_label
                logger.info(f"Processed {app_label}: {results}")
                return jsonify(results), status

            elif app_label == "rdf50_m1.0_fsk_f2.0":
                print(model_path, scaler_path)
                results, status = fsk_answers_v2(
                    answers, model_path=model_path, scaler_path=scaler_path)
                results['application_label'] = app_label
                logger.info(f"Processed {app_label}: {results}")
                return jsonify(results), status

            elif app_label == "rdf50_m1.0_fk_f1.0":
                print(model_path, scaler_path)
                results, status = fsk_answers_v2(
                    answers, model_path=model_path, scaler_path=scaler_path)
                results['application_label'] = app_label
                logger.info(f"Processed {app_label}: {results}")
                return jsonify(results), status

        except Exception as e:
            logger.error(f"Error in /eval: {str(e)}")
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

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

            if not isinstance(answers, dict):
                logger.error(f"Invalid answers format: {type(answers)}")
                return jsonify({"error": "Answers must be a dictionary"}), 400
            if not answers:
                logger.error("Answers dictionary is empty")
                return jsonify({"error": "Answers dictionary is empty"}), 400

            if application_label == "rdf50_m1.0_fsk_f1.0":
                result, status = fsk_answers_v1(
                    answers, model_path=model_path, scaler_path=scaler_path)
            else:
                logger.error(
                    f"Unsupported application_label: {application_label}")
                return jsonify({"error": f"Unsupported application_label: {application_label}"}), 400

            result['application_label'] = application_label
            logger.info(
                f"Prediction successful for application_label: {application_label}")
            return jsonify(result), status
        except Exception as e:
            logger.error(f"Error in /predict: {str(e)}", exc_info=True)
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

    @app.route('/lucis', methods=['POST'])
    def lucis():
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config = ModelConfig(scoring='roc_auc')

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

            scale_clean_engineer_dat, scaler, selected_features = preprocess_and_select_features(
                raw_dat)
            if scale_clean_engineer_dat is None or scaler is None or not selected_features:
                logger.error("Preprocessing or feature selection failed")
                return jsonify({"error": "Preprocessing or feature selection failed"}), 500
            logger.info({"message": "Selected features",
                        "features": selected_features})

            model, metrics, final_features = train_and_evaluate_model(
                scale_clean_engineer_dat, selected_features)
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

            latest_metadata = load_latest_model_metadata(paths['model'][0])
            latest_cv_recall = latest_metadata.get('cv_recall', 0.0)
            metadata_exists = bool(latest_metadata.get('model_path'))
            logger.info({
                "message": "Model comparison",
                "metadata_exists": metadata_exists,
                "current_cv_roc_auc": metrics_summary['cv_roc_auc'],
                "latest_cv_recall": latest_cv_recall
            })

            artifact_info = {
                'scaler_path': '', 'scaler_checksum': '',
                'model_path': '', 'model_checksum': '',
                'metadata_path': '', 'metadata_checksum': ''
            }
            if not metadata_exists or (metrics_summary['cv_recall'] > latest_cv_recall and metrics_summary['cv_score_std'] < 0.10):
                logger.info({
                    "message": "Saving artifacts",
                    "first_run": not metadata_exists,
                    "cv_recall": metrics_summary['cv_recall'],
                    "latest_cv_recall": latest_cv_recall
                })
                artifact_info = save_all_artifacts(
                    scaler, model, importance_df, final_features, metrics_summary, paths, timestamp
                )
                if artifact_info is None:
                    logger.error("Failed to save artifacts")
                    return jsonify({"error": "Failed to save artifacts"}), 500

            package_versions = {
                'sklearn': sklearn.__version__,
                'joblib': joblib.__version__
            }
            scaler_instructions = get_scaler_instructions(
                artifact_info, final_features, package_versions)

            final_features_dict = [
                {'feature': row['feature'],
                    'importance': float(row['importance'])}
                for _, row in importance_df.iterrows()
            ]

            response = {
                '0.model': str(model),
                '1.latest_cv_recall': latest_cv_recall,
                '2.metrics': {
                    'cv_roc_auc': metrics_summary['cv_roc_auc'],
                    'cv_accuracy': metrics_summary['cv_accuracy'],
                    'cv_precision': metrics_summary['cv_precision'],
                    'cv_recall': metrics_summary['cv_recall'],
                    'cv_f1': metrics_summary['cv_f1'],
                    'test_roc_auc': metrics_summary['test_roc_auc'],
                    'test_accuracy': metrics_summary['test_accuracy'],
                    'test_precision': metrics_summary['test_precision'],
                    'test_recall': metrics_summary['test_recall'],
                    'test_f1': metrics_summary['test_f1'],
                    'cv_score_std': metrics_summary['cv_score_std'],
                    'overfitting_gap': metrics_summary['overfitting_gap'],
                },
                '3.confidence_intervals': {
                    'accuracy': {
                        'lower': metrics_summary['accuracy_ci_lower'],
                        'upper': metrics_summary['accuracy_ci_upper']
                    },
                    'precision': {
                        'lower': metrics_summary['precision_ci_lower'],
                        'upper': metrics_summary['precision_ci_upper']
                    },
                    'recall': {
                        'lower': metrics_summary['recall_ci_lower'],
                        'upper': metrics_summary['recall_ci_upper']
                    },
                    'f1': {
                        'lower': metrics_summary['f1_ci_lower'],
                        'upper': metrics_summary['f1_ci_upper']
                    },
                    'roc_auc': {
                        'lower': metrics_summary['roc_auc_ci_lower'],
                        'upper': metrics_summary['roc_auc_ci_upper']
                    }
                },
                '4.final_features': final_features_dict,
                '5.artifacts': artifact_info,
                '6.package_versions': package_versions,
                '7.scaler_instructions': scaler_instructions,
            }

            logger.info(
                {"message": "Training completed successfully", "response": response})
            return jsonify(response), 200

        except ValueError as ve:
            logger.error({"message": "ValueError in lumen", "error": str(ve)})
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except (OSError, PermissionError) as e:
            logger.error({"message": "System Error in lumen", "error": str(e)})
            return jsonify({"error": f"System Error: {str(e)}"}), 500
        except Exception as e:
            logger.error(
                {"message": "Internal Server Error in lumen", "error": str(e)})
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
