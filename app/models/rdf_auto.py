import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE  
from joblib import dump
from app.data.data_loading import data_loading_fsk_v1
from app.data.data_cleaning import data_cleaning_fsk_v1
from app.data.data_transforming import data_transforming_fsk_v1
from app.data.data_engineering import data_engineering_fsk_v1
from app.data.data_preprocessing import data_preprocessing
from app.models.correlation import compute_correlations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
        
        if 'Spearman' not in corr_dat.columns:
            logger.error("Column 'Spearman' not found in DataFrame.")
            return []
        
        sorted_corr = corr_dat.sort_values(by='Spearman', ascending=False)
        top_positive = sorted_corr[sorted_corr['Spearman'] > 0].head(n).index.tolist()
        top_negative = sorted_corr[sorted_corr['Spearman'] < 0].tail(n).index.tolist()
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
            'importance': model.feature_importances
        }).sort_values('importance', ascending=False)
        logger.info("Feature Importance calculated successfully.")
        logger.info(f"\n{feature_importance.to_string()}")
        return feature_importance
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        return pd.DataFrame()

def rfe_feature_selection(X: pd.DataFrame, y: pd.Series, target_col: str = 'ust', scoring: str = 'roc_auc') -> Tuple[list, RandomForestClassifier, dict]:
    try:
        current_features = list(X.columns)
        best_features = current_features.copy()
        best_model = None
        best_metrics = None
        best_score = -float('inf')

        logger.info(f"Starting RFE feature selection with {len(current_features)} features: {current_features}")

        n_samples = len(y)
        n_folds = min(5, n_samples)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        logger.info(f"Using StratifiedKFold with {n_folds} folds")

        max_depth = 5 if n_samples <= 100 else 8
        min_samples_split = 10 if n_samples <= 100 else 5
        n_features_to_select = max(3, len(current_features) // 2)

        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=42
        )

        smote = SMOTE(random_state=42, k_neighbors=3)
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"Applied SMOTE: Resampled data shape: {X_resampled.shape}")
            check_class_balance(y_resampled, target_col)
            X = X_resampled
            y = y_resampled
        except ValueError as e:
            logger.warning(f"SMOTE failed: {str(e)}. Proceeding with original data.")

        rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_].tolist()
        logger.info(f"RFE selected {len(selected_features)} features: {selected_features}")

        X_selected = X[selected_features]
        model.fit(X_selected, y)

        cv_scores = []
        for train_idx, test_idx in cv.split(X_selected, y):
            X_train_fold, X_test_fold = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            class_counts = pd.Series(y_train_fold).value_counts()
            logger.info(f"Fold class balance: {class_counts.to_dict()}")
            if len(class_counts) < 2:
                logger.warning(f"Only one class in fold training set. Skipping ROC AUC for this fold.")
                cv_scores.append(np.nan)
                continue
            model.fit(X_train_fold, y_train_fold)
            y_prob_fold = model.predict_proba(X_test_fold)[:, 1]
            try:
                score = roc_auc_score(y_test_fold, y_prob_fold)
                cv_scores.append(score)
            except ValueError:
                logger.warning(f"ROC AUC undefined for this fold due to single class in y_test_fold.")
                cv_scores.append(np.nan)

        cv_score = np.nanmean(cv_scores) if cv_scores else np.nan
        logger.info(f"Cross-validated {scoring} with selected features: {cv_score:.3f}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan

        y_pred_cv = cross_val_predict(model, X_selected, y, cv=cv)
        y_prob_cv = cross_val_predict(model, X_selected, y, cv=cv, method='predict_proba')
        cv_accuracy = accuracy_score(y, y_pred_cv)
        cv_precision = precision_score(y, y_pred_cv, zero_division=1)
        cv_recall = recall_score(y, y_pred_cv, zero_division=1)
        cv_f1 = f1_score(y, y_pred_cv, zero_division=1)
        cv_roc_auc = roc_auc_score(y, y_prob_cv[:, 1]) if len(np.unique(y)) > 1 else np.nan

        feature_importance = features_importance(model, X_selected)
        best_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=1),
            'f1': f1_score(y_test, y_pred, zero_division=1),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'cv_scores': cv_scores,
            'cross_validated_accuracy': cv_accuracy,
            'cross_validated_precision': cv_precision,
            'cross_validated_recall': cv_recall,
            'cross_validated_f1': cv_f1,
            'cross_validated_roc_auc': cv_roc_auc,
            'test_roc_auc': test_roc_auc,
            'feature_importance': feature_importance.to_dict(),
            'best_features': selected_features
        }

        best_features = selected_features
        best_model = model
        best_score = cv_score
        logger.info(f"Best model found with cross-validated {scoring}: {best_score:.3f}, Features: {best_features}")

        return best_features, best_model, best_metrics

    except Exception as e:
        logger.error(f"Error during RFE feature selection: {str(e)}")
        return [], None, {}

def train_model(scale_clean_engineer_dat: pd.DataFrame, selected_features: list, target_col: str = 'ust') -> Tuple[Optional[RandomForestClassifier], Optional[dict]]:
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
        
        logger.info("Starting RFE feature selection...")
        best_features, best_model, best_metrics = rfe_feature_selection(X, y, target_col, scoring='roc_auc')
        
        if not best_features or best_model is None or best_metrics is None:
            logger.error("RFE feature selection failed to find a suitable model.")
            return None, None

        logger.info(f"Best features after RFE: {best_features}")

        logger.info("Final Model Metrics:")
        logger.info(f"Accuracy (Test Set): {best_metrics['accuracy']:.3f}")
        logger.info(f"Precision (Test Set): {best_metrics['precision']:.3f}")
        logger.info(f"F1 Score (Test Set): {best_metrics['f1']:.3f}")
        logger.info(f"Cross-Validated ROC AUC: {best_metrics['cross_validated_roc_auc']:.3f}")
        logger.info(f"Cross-Validated Accuracy: {best_metrics['cross_validated_accuracy']:.3f}")
        logger.info(f"Cross-Validated Precision: {best_metrics['cross_validated_precision']:.3f}")
        logger.info(f"Cross-Validated Recall: {best_metrics['cross_validated_recall']:.3f}")
        logger.info(f"Cross-Validated F1 Score: {best_metrics['cross_validated_f1']:.3f}")
        logger.info(f"Test ROC AUC: {best_metrics['test_roc_auc']:.3f}")
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
        os.makedirs(scaler_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
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
    final_model_filename = model_filename if model_filename else f"custom_random_forest_model_{timestamp}"

    logger.info("Starting random forest pipeline...")
    raw_dat = data_loading_fsk_v1()
    if raw_dat is not None:
        logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        if cleaned_dat is not None:
            logger.info(f"Cleaned DataFrame Shape: {cleaned_dat.shape}")
            transformed_dat = data_transforming_fsk_v1(cleaned_dat)
            if transformed_dat is not None:
                logger.info(f"Transformed DataFrame Shape: {transformed_dat.shape}")
                engineer_dat = data_engineering_fsk_v1(transformed_dat)
                if engineer_dat is not None:
                    logger.info(f"Features Engineered DataFrame Shape: {engineer_dat.shape}")
                    result = data_preprocessing(engineer_dat)
                    if result is not None:
                        scale_clean_engineer_dat, scaler = result
                        logger.info(f"Cleaned Engineered DataFrame Shape (After Outlier Removal): {scale_clean_engineer_dat.shape}")
                        corr_dat = compute_correlations(scale_clean_engineer_dat)
                        scaler_path = os.path.join(scaler_dir, f"{final_scaler_filename}.pkl")
                        dump(scaler, scaler_path)
                        logger.info(f"Saved scaler to {scaler_path}")
                        if corr_dat is not None:
                            logger.info("Correlations with ust (Pearson and Spearman):")
                            logger.info(f"\n{corr_dat}")
                            selected_features = select_top_features(corr_dat, n=10)
                            if selected_features:
                                model, metrics = train_model(scale_clean_engineer_dat, selected_features)
                                if model is not None and metrics is not None:
                                    logger.info("Train random forest model complete")
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
                                        'test_roc_auc': [metrics['test_roc_auc']],
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
                                    logger.error("Failed to train random forest model.")
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