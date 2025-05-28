import logging
import os
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score, f1_score
from joblib import dump
from app.data.data_loading import data_loading_fsk_v1
from app.data.data_cleaning import data_cleaning_fsk_v1
from app.data.data_transforming import data_transform_fsk_v1
from app.data.data_engineering import data_engineer_fsk_v1
from app.data.data_preprocessing import data_preprocessing
from app.models.correlation import compute_correlations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_class_balance(y: pd.Series, target_col: str) -> None:
    try:
        class_counts = y.value_counts()
        total_samples = len(y)
        class_balance = class_counts / total_samples * 100
        logger.info(f"Class balance for '{target_col}':")
        for class_label, percentage in class_balance.items():
            logger.info(f"Class {class_label}: {class_counts[class_label]} samples ({percentage:.2f}%)")
    except Exception as e:
        logger.error(f"Error checking class balance: {str(e)}")

def select_top_features(corr_dat: pd.DataFrame, n: int = 10) -> list:
    try:
        if corr_dat is None or corr_dat.empty:
            logger.error("Correlation DataFrame is None or empty.")
            return []

        top_positive = corr_dat[corr_dat['Pearson'] > 0].head(n).index.tolist()
        top_negative = corr_dat[corr_dat['Pearson'] < 0].tail(n).index.tolist()
        selected_features = top_positive + top_negative
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return selected_features
    except Exception as e:
        logger.error(f"Error selecting features: {str(e)}")
        return []

def features_importance(model, X, y=None) -> pd.DataFrame:
    try:
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(model.coef_[0])  
        }).sort_values('importance', ascending=False)
        logger.info("Feature Importance calculated successfully.")
        logger.info(f"\n{feature_importance.to_string()}")
        return feature_importance
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        return pd.DataFrame()

def backward_elimination(X: pd.DataFrame, y: pd.Series, target_col: str = 'ust', scoring: str = 'roc_auc') -> Tuple[list, LogisticRegression, dict]:
   
    try:
        
        current_features = list(X.columns)
        best_features = current_features.copy()
        best_model = None
        best_metrics = None
        best_score = -float('inf')  

        logger.info(f"Starting backward elimination with {len(current_features)} features: {current_features}")

        while current_features:
            
            X_current = X[current_features]
            X_train, X_test, y_train, y_test = train_test_split(
                X_current, y, test_size=0.2, random_state=42, stratify=y
            )

            
            model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_train, y_train)

            
            y_prob = model.predict_proba(X_test)[:, 1]
            current_score = roc_auc_score(y_test, y_prob)
            logger.info(f"Current features: {current_features}, ROC AUC: {current_score:.3f}")

            
            if current_score > best_score:
                best_score = current_score
                best_features = current_features.copy()
                best_model = model

                
                y_pred_cv = cross_val_predict(model, X_current, y, cv=5)
                y_prob_cv = cross_val_predict(model, X_current, y, cv=5, method='predict_proba')
                cv_accuracy = accuracy_score(y, y_pred_cv)
                cv_precision = precision_score(y, y_pred_cv, zero_division=1)
                cv_recall = recall_score(y, y_pred_cv, zero_division=1)
                cv_f1 = f1_score(y, y_pred_cv, zero_division=1)
                cv_roc_auc = roc_auc_score(y, y_prob_cv[:, 1])

                y_pred = model.predict(X_test)
                feature_importance = features_importance(model, X_current)
                best_metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=1),
                    'f1': f1_score(y_test, y_pred, zero_division=1),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'cv_scores': cross_val_score(model, X_current, y, cv=5, scoring=scoring).tolist(),
                    'cross_validated_accuracy': cv_accuracy,
                    'cross_validated_precision': cv_precision,
                    'cross_validated_recall': cv_recall,
                    'cross_validated_f1': cv_f1,
                    'cross_validated_roc_auc': cv_roc_auc,
                    'feature_importance': feature_importance.to_dict()
                }
                logger.info(f"New best model found with ROC AUC: {best_score:.3f}, Features: {best_features}")

            
            if len(current_features) == 1:
                break

            
            feature_importance = features_importance(model, X_current)
            least_important_feature = feature_importance.iloc[-1]['feature']
            current_features.remove(least_important_feature)
            logger.info(f"Removed least important feature: {least_important_feature}, Remaining features: {current_features}")

            
            new_X = X[current_features]
            X_train, X_test, y_train, y_test = train_test_split(
                new_X, y, test_size=0.2, random_state=42, stratify=y
            )
            model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            new_score = roc_auc_score(y_test, y_prob)

            if new_score < best_score:
                logger.info(f"Stopping backward elimination. ROC AUC decreased to {new_score:.3f} after removing {least_important_feature}")
                break

        return best_features, best_model, best_metrics
    except Exception as e:
        logger.error(f"Error during backward elimination: {str(e)}")
        return [], None, None

def train_logistic_model(scale_clean_engineer_dat: pd.DataFrame, selected_features: list, target_col: str = 'ust') -> Tuple[Optional[LogisticRegression], Optional[dict]]:
    try:
        if scale_clean_engineer_dat is None or scale_clean_engineer_dat.empty:
            logger.error("Input DataFrame is None or empty.")
            return None, None

        if not selected_features:
            logger.error("No features selected for training.")
            return None, None

        if target_col not in scale_clean_engineer_dat.columns:
            logger.error(f"Target column '{target_col}' not found in DataFrame.")
            return None, None

        X = scale_clean_engineer_dat[selected_features]
        y = scale_clean_engineer_dat[target_col]

        if y.nunique() != 2:
            logger.error(f"Target column '{target_col}' is not binary. Found {y.nunique()} unique values.")
            return None, None

        check_class_balance(y, target_col)


        logger.info("Starting backward elimination for feature selection...")
        best_features, best_model, best_metrics = backward_elimination(X, y, target_col, scoring='roc_auc')
        
        if not best_features or best_model is None or best_metrics is None:
            logger.error("Backward elimination failed to find a suitable model.")
            return None, None

        logger.info(f"Best features after backward elimination: {best_features}")

        
        logger.info("Final Model Metrics:")
        logger.info(f"Accuracy (Test Set): {best_metrics['accuracy']:.3f}")
        logger.info(f"Precision (Test Set): {best_metrics['precision']:.3f}")
        logger.info(f"F1 Score (Test Set): {best_metrics['f1']:.3f}")
        logger.info(f"Cross-Validated ROC AUC: {best_metrics['cross_validated_roc_auc']:.3f}")
        logger.info(f"Cross-Validated Accuracy: {best_metrics['cross_validated_accuracy']:.3f}")
        logger.info(f"Cross-Validated Precision: {best_metrics['cross_validated_precision']:.3f}")
        logger.info(f"Cross-Validated Recall: {best_metrics['cross_validated_recall']:.3f}")
        logger.info(f"Cross-Validated F1 Score: {best_metrics['cross_validated_f1']:.3f}")
        logger.info(f"Classification Report (Test Set):\n{pd.DataFrame(best_metrics['classification_report']).to_string()}")

        return best_model, best_metrics
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return None, None

if __name__ == "__main__":
    
    output_dir = "output_data"
    scaler_dir = "save_scaler"
    output_dir = "output_data"
    model_dir = "save_models"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        output_dir = "output_data"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(scaler_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directories {output_dir} or {model_dir}: {e}")
        exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    scaler_filename = None
    metrics_filename = None
    feature_importance_filename = None
    model_filename = None
    
    final_scaler_filename = scaler_filename if scaler_filename else f"custom_scaler_{timestamp}"
    final_metrics_filename = metrics_filename if metrics_filename else f"custom_metrics_{timestamp}"
    final_feature_importance_filename = feature_importance_filename if feature_importance_filename else f"custom_feature_importance_{timestamp}"
    final_model_filename = model_filename if model_filename else f"custom_logistic_model_{timestamp}"

    logger.info("Starting logistic regression pipeline...")
    raw_dat = data_loading_fsk_v1()
    if raw_dat is not None:
        logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        if cleaned_dat is not None:
            logger.info(f"Cleaned DataFrame Shape: {cleaned_dat.shape}")
            transformed_dat = data_transform_fsk_v1(cleaned_dat)
            if transformed_dat is not None:
                logger.info(f"Transformed DataFrame Shape: {transformed_dat.shape}")
                engineer_dat = data_engineer_fsk_v1(transformed_dat)
                if engineer_dat is not None:
                    logger.info(f"Features Engineered DataFrame Shape: {engineer_dat.shape}")
                    result = data_preprocessing(engineer_dat)
                    if result is not None:
                        scale_clean_engineer_dat, scaler = result
                        logger.info(f"Cleaned Engineered DataFrame Shape (After Outlier Removal): {scale_clean_engineer_dat.shape}")
                        scaler_path = os.path.join(scaler_dir, f"{final_scaler_filename}.pkl")
                        dump(scaler, scaler_path)
                        logger.info(f"Saved scaler to {scaler_path}")
                        corr_dat = compute_correlations(scale_clean_engineer_dat)
                        if corr_dat is not None:
                            logger.info("Correlations with ust (Pearson and Spearman):")
                            logger.info(f"\n{corr_dat}")
                            selected_features = select_top_features(corr_dat, n=10)
                            if selected_features:
                                model, metrics = train_logistic_model(scale_clean_engineer_dat, selected_features)
                                if model is not None and metrics is not None:
                                    logger.info("Train logistic regression model complete")
                                    metrics_df = pd.DataFrame({
                                        'accuracy': [metrics['accuracy']],
                                        'precision': [metrics['precision']],
                                        'f1': [metrics['f1']],
                                        'precision_0': [metrics['classification_report']['0']['precision']],
                                        'recall_0': [metrics['classification_report']['0']['recall']],
                                        'f1_0': [metrics['classification_report']['0']['f1-score']],
                                        'precision_1': [metrics['classification_report']['1']['precision']],
                                        'recall_1': [metrics['classification_report']['1']['recall']],
                                        'f1_1': [metrics['classification_report']['1']['f1-score']],
                                        'mean_cv_accuracy': [np.mean(metrics['cv_scores'])],
                                        'std_cv_accuracy': [np.std(metrics['cv_scores'])],
                                        'cross_validated_accuracy_mean': [metrics.get('cross_validated_ACCURACY_mean', np.mean(metrics['cv_scores']))],
                                        'cross_validated_accuracy_all_folds': [metrics['cv_scores']],
                                        'cross_validated_accuracy': [metrics['cross_validated_accuracy']],
                                        'cross_validated_precision': [metrics['cross_validated_precision']],
                                        'cross_validated_recall': [metrics['cross_validated_recall']],
                                        'cross_validated_f1': [metrics['cross_validated_f1']],
                                        'cross_validated_roc_auc': [metrics['cross_validated_roc_auc']],
                                    })
                                    metrics_path = os.path.join(output_dir, f"{final_metrics_filename}.csv")
                                    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
                                    logger.info(f"Saved model metrics to {metrics_path}")

                                    feature_importance_df = pd.DataFrame(metrics['feature_importance'])
                                    feature_importance_path = os.path.join(output_dir, f"{final_feature_importance_filename}.csv")
                                    feature_importance_df.to_csv(feature_importance_path, index=False, encoding='utf-8-sig')
                                    logger.info(f"Saved feature importance to {feature_importance_path}")

                                    model_path = os.path.join(model_dir, f"{final_model_filename}.pkl")
                                    dump(model, model_path)
                                    logger.info(f"Saved model to {model_path}")
                                else:
                                    logger.error("Failed to train logistic regression model.")
                            else:
                                logger.error("No features selected for training.")
                        else:
                            logger.error("Failed to compute correlations.")
                    else:
                        logger.error("Failed to remove outliers.")
                else:
                    logger.error("Failed to engineer features.")
            else:
                logger.error("Failed to transform data.")
        else:
            logger.error("Failed to clean data.")
    else:
        logger.error("Failed to load data.")