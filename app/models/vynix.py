import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from joblib import dump, load
from statsmodels.stats.outliers_influence import variance_inflation_factor  # เพิ่มสำหรับ VIF
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

class ModelConfig:
    def __init__(self, scoring: str = 'roc_auc', oversampling_method: str = 'smote'):
        self.n_estimators = [100, 200]
        self.max_depth = [3, 5, None]
        self.min_samples_split = [5, 10]
        self.n_features_to_select = 10  
        self.n_folds = 5
        self.scoring = scoring
        self.random_state = 42
        self.oversampling_method = oversampling_method
        self.correlation_threshold = 0.7  
        self.vif_threshold = 5.0  

def validate_dataframe(df: pd.DataFrame, name: str) -> bool:
    if df is None or df.empty:
        logger.error(f"{name} DataFrame is None or empty.")
        return False
    return True

def compute_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """คำนวณ Correlation Matrix และคืน DataFrame ของค่าสหสัมพันธ์"""
    try:
        corr_matrix = X.corr(method='pearson')
        logger.info("Computed correlation matrix successfully.")
        return corr_matrix
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {str(e)}")
        return pd.DataFrame()

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """คำนวณ Variance Inflation Factor (VIF) สำหรับแต่ละฟีเจอร์"""
    try:
        
        X_vif = X.copy()
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X_vif.columns
        vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        logger.info("Computed VIF successfully.")
        return vif_data
    except Exception as e:
        logger.error(f"Error computing VIF: {str(e)}")
        return pd.DataFrame()

def filter_multicollinear_features(X: pd.DataFrame, corr_matrix: pd.DataFrame, 
                                  config: ModelConfig) -> List[str]:
    
    try:
        if corr_matrix.empty:
            logger.warning("Correlation matrix is empty. Returning all features.")
            return X.columns.tolist()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > config.correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                          corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            logger.warning(f"Found {len(high_corr_pairs)} high correlation pairs:")
            for pair in high_corr_pairs:
                logger.info(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
        # เก็บฟีเจอร์ที่มี correlation สูงไว้ แล้วเลือกตัวที่มี variance สูงกว่า
        features_to_remove = set()
        for feat1, feat2, _ in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                var1 = X[feat1].var()
                var2 = X[feat2].var()
                # ลบฟีเจอร์ที่มี variance ต่ำกว่า
                features_to_remove.add(feat2 if var1 > var2 else feat1)
        
        selected_features = [f for f in X.columns if f not in features_to_remove]
        logger.info(f"Filtered features: {selected_features}")
        return selected_features
    except Exception as e:
        logger.error(f"Error filtering multicollinear features: {str(e)}")
        return X.columns.tolist()

def select_top_features(corr_dat: pd.DataFrame, X: pd.DataFrame, n: int = 10, 
                       config: ModelConfig = None) -> List[str]:
    
    try:
        if not validate_dataframe(corr_dat, "Correlation") or not validate_dataframe(X, "Features"):
            return []
        if 'Spearman' not in corr_dat.columns or corr_dat['Spearman'].isna().all():
            logger.error("Invalid or missing 'Spearman' column.")
            return []
        
        # เลือกฟีเจอร์ที่มี Spearman correlation สูง
        sorted_corr = corr_dat.sort_values(by='Spearman', ascending=False)
        top_positive = sorted_corr[sorted_corr['Spearman'] > 0].head(n).index.tolist()
        top_negative = sorted_corr[sorted_corr['Spearman'] < 0].tail(n).index.tolist()
        candidate_features = top_positive + top_negative
        
        # คำนวณ correlation matrix และกรองฟีเจอร์ที่มี multicollinearity
        X_candidate = X[candidate_features]
        corr_matrix = compute_correlation_matrix(X_candidate)
        selected_features = filter_multicollinear_features(X_candidate, corr_matrix, config)
        
        # จำกัดจำนวนฟีเจอร์ตาม n
        selected_features = selected_features[:n]
        logger.info(f"Selected {len(selected_features)} features after multicollinearity check: {selected_features}")
        return selected_features
    except KeyError as e:
        logger.error(f"KeyError in select_top_features: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in select_top_features: {str(e)}")
        return []

def features_importance(model: RandomForestClassifier, X: pd.DataFrame) -> pd.DataFrame:
    try:
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        logger.info("Feature Importance:\n" + feature_importance.to_string())
        return feature_importance
    except AttributeError as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        return pd.DataFrame()

def compute_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, 
                                n_bootstraps: int = 1000, alpha: float = 0.05) -> Dict:
    try:
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        n_samples = len(y_true)
        if n_samples < 10:
            logger.warning("Test set too small for reliable CI calculation (< 10 samples)")
            ci_dict = {}
            for m in metrics:
                ci_dict[f"{m}_ci_lower"] = 0.0
                ci_dict[f"{m}_ci_upper"] = 0.0
            return ci_dict
        
        if len(np.unique(y_true)) < 2:
            logger.warning("Test set has only one class, CI cannot be computed")
            ci_dict = {}
            for m in metrics:
                ci_dict[f"{m}_ci_lower"] = 0.0
                ci_dict[f"{m}_ci_upper"] = 0.0
            return ci_dict
        
        rng = np.random.default_rng(seed=42)
        
        for _ in range(n_bootstraps):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_prob_boot = y_prob[indices]
            
            if len(np.unique(y_true_boot)) < 2:
                continue
                
            metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics['roc_auc'].append(roc_auc_score(y_true_boot, y_prob_boot) if len(np.unique(y_true_boot)) > 1 else np.nan)
        
        ci_results = {}
        for metric, values in metrics.items():
            values = np.array([v for v in values if not np.isnan(v)])
            if len(values) < 10:  # Ensure enough bootstrap samples
                logger.warning(f"Too few valid bootstrap samples for {metric} ({len(values)}), setting CI to 0.0")
                ci_results[f'{metric}_ci_lower'] = 0.0
                ci_results[f'{metric}_ci_upper'] = 0.0
            else:
                ci_lower = np.nanpercentile(values, alpha / 2 * 100)
                ci_upper = np.nanpercentile(values, 100 - (alpha / 2 * 100))
                if ci_lower == ci_upper:
                    logger.warning(f"CI for {metric} is identical ({ci_lower}), adjusting slightly")
                    ci_lower = max(0.0, ci_lower - 0.01)
                    ci_upper = min(1.0, ci_upper + 0.01)
                ci_results[f'{metric}_ci_lower'] = ci_lower
                ci_results[f'{metric}_ci_upper'] = ci_upper
                
        return ci_results
    except Exception as e:
        logger.error(f"Error computing confidence intervals: {str(e)}")
        ci_dict = {}
        for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            ci_dict[f"{m}_ci_lower"] = 0.0
            ci_dict[f"{m}_ci_upper"] = 0.0
        return ci_dict

def evaluate_model(model: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series, 
                  X_test: pd.DataFrame, y_test: pd.Series, cv: StratifiedKFold) -> Dict:
    try:
        # Cross-validation scores
        cv_scores = []
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
            model.fit(X_train_fold, y_train_fold)
            y_prob_fold = model.predict_proba(X_test_fold)[:, 1]
            if len(np.unique(y_test_fold)) > 1:
                cv_scores.append(roc_auc_score(y_test_fold, y_prob_fold))
            else:
                cv_scores.append(np.nan)

        # Cross-validation metrics
        y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
        y_prob_cv = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')
        cv_metrics = {
            'cross_validated_accuracy': accuracy_score(y_train, y_pred_cv),
            'cross_validated_precision': precision_score(y_train, y_pred_cv, zero_division=1),
            'cross_validated_recall': recall_score(y_train, y_pred_cv, zero_division=1),
            'cross_validated_f1': f1_score(y_train, y_pred_cv, zero_division=1),
            'cross_validated_roc_auc': roc_auc_score(y_train, y_prob_cv[:, 1]) if len(np.unique(y_train)) > 1 else np.nan,
            'cv_scores': cv_scores,
            'cv_score_mean': np.nanmean(cv_scores) if cv_scores else np.nan,
            'cv_score_std': np.nanstd(cv_scores) if cv_scores else np.nan
        }

        if cv_metrics['cv_score_std'] > 0.1:
            logger.warning(f"High variance in CV scores (std: {cv_metrics['cv_score_std']:.3f}). Model may be unstable.")

        # Test set metrics
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=1),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=1),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=1),
            'test_roc_auc': roc_auc_score(y_test, y_prob_test) if len(np.unique(y_test)) > 1 else np.nan
        }

        ci_metrics = compute_confidence_intervals(y_test.values, y_pred_test, y_prob_test)

        if abs(cv_metrics['cross_validated_accuracy'] - test_metrics['test_accuracy']) > 0.1:
            logger.warning("Possible overfitting detected: significant performance gap between CV and test set.")

        return {**cv_metrics, **test_metrics, **ci_metrics}
    except ValueError as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in evaluate_model: {str(e)}")
        return {}

def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> RandomForestClassifier:
    try:
        param_grid = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'min_samples_split': config.min_samples_split
        }
        base_model = RandomForestClassifier(random_state=config.random_state, class_weight='balanced')
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=min(config.n_folds, len(y)),
            scoring=config.scoring,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except ValueError as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        return RandomForestClassifier(random_state=config.random_state, class_weight='balanced')
    except Exception as e:
        logger.error(f"Unexpected error in hyperparameter tuning: {str(e)}")
        return RandomForestClassifier(random_state=config.random_state, class_weight='balanced')

def rfe_feature_selection(X: pd.DataFrame, y: pd.Series, config: ModelConfig, target_col: str = 'ust') -> Tuple[List[str], Optional[RandomForestClassifier], Dict]:
    try:
        if not validate_dataframe(X, "Feature") or y is None or y.empty:
            return [], None, {}

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.random_state, stratify=y
        )

        # Apply oversampling on training data only
        X_train, y_train = apply_oversampling(X_train, y_train, config.oversampling_method, config.random_state)
        check_class_balance(y_train, target_col)

        # Tune hyperparameters
        model = tune_hyperparameters(X_train, y_train, config)

        # RFE feature selection
        n_features = max(config.n_features_to_select, len(X.columns) // 2)
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit(X_train, y_train)
        selected_features = X.columns[rfe.support_].tolist()
        logger.info(f"RFE selected {len(selected_features)} features: {selected_features}")

        # Train final model with selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        model.fit(X_train_selected, y_train)

        cv = StratifiedKFold(n_splits=min(config.n_folds, len(y_train)), shuffle=True, random_state=config.random_state)
        metrics = evaluate_model(model, X_train_selected, y_train, X_test_selected, y_test, cv)

        metrics['best_features'] = selected_features
        metrics['feature_importance'] = features_importance(model, X_train_selected).to_dict()
        metrics['model_params'] = model.get_params()
        metrics['oversampling_method'] = config.oversampling_method
        metrics['n_features_selected'] = len(selected_features)

        return selected_features, model, metrics

    except Exception as e:
        logger.error(f"Error during RFE feature selection: {str(e)}")
        return [], None, {}

def train_model(scale_clean_engineer_dat: pd.DataFrame, selected_features: List[str], 
                target_col: str = 'ust', scoring: str = 'roc_auc') -> Tuple[Optional[RandomForestClassifier], Optional[Dict]]:
    try:
        if not validate_dataframe(scale_clean_engineer_dat, "Input") or not selected_features:
            return None, None

        if target_col not in scale_clean_engineer_dat.columns:
            logger.error(f"Target column '{target_col}' not found.")
            return None, None

        X = scale_clean_engineer_dat[selected_features]
        y = scale_clean_engineer_dat[target_col]

        if y.nunique() != 2:
            logger.error(f"Target column '{target_col}' is not binary. Found {y.nunique()} unique values.")
            return None, None

        check_class_balance(y, target_col)
        config = ModelConfig(scoring=scoring)
        best_features, best_model, best_metrics = rfe_feature_selection(X, y, config, target_col)

        if not best_features or best_model is None:
            logger.error("RFE feature selection failed.")
            return None, None

        logger.info(f"Final Model Metrics: {best_metrics}")
        return best_model, best_metrics

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return None, None

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    config = ModelConfig()
    output_dir = "output_data"
    scaler_dir = "save_scaler"
    model_dir = "save_models"

    for directory in [output_dir, scaler_dir, model_dir]:
        os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_scaler_filename = f"custom_scaler_{timestamp}.pkl"
    final_metrics_filename = f"custom_metrics_{timestamp}.csv"
    final_feature_importance_filename = f"custom_feature_importance_{timestamp}.csv"
    final_model_filename = f"custom_random_forest_model_{timestamp}.pkl"
    final_correlation_filename = f"correlation_matrix_{timestamp}.csv"  # เพิ่มสำหรับบันทึก correlation
    final_vif_filename = f"vif_scores_{timestamp}.csv"  # เพิ่มสำหรับบันทึก VIF

    logger.info("Starting random forest pipeline...")
    raw_dat = data_loading_fsk_v1()
    if validate_dataframe(raw_dat, "Raw"):
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        if validate_dataframe(cleaned_dat, "Cleaned"):
            transformed_dat = data_transforming_fsk_v1(cleaned_dat)
            if validate_dataframe(transformed_dat, "Transformed"):
                engineer_dat = data_engineering_fsk_v1(transformed_dat)
                if validate_dataframe(engineer_dat, "Engineered"):
                    result = data_preprocessing(engineer_dat)
                    if result is not None:
                        scale_clean_engineer_dat, scaler = result
                        corr_dat = compute_correlations(scale_clean_engineer_dat)
                        scaler_path = os.path.join(scaler_dir, final_scaler_filename)
                        dump(scaler, scaler_path)
                        logger.info(f"Saved scaler to {scaler_path}")
                        if validate_dataframe(corr_dat, "Correlation"):
                            # คำนวณ Correlation Matrix และ VIF
                            X = scale_clean_engineer_dat.drop(columns=['ust'], errors='ignore')
                            corr_matrix = compute_correlation_matrix(X)
                            vif_data = compute_vif(X)
                            
                            # บันทึก Correlation Matrix และ VIF
                            if not corr_matrix.empty:
                                corr_path = os.path.join(output_dir, final_correlation_filename)
                                corr_matrix.to_csv(corr_path, encoding='utf-8-sig')
                                logger.info(f"Saved correlation matrix to {corr_path}")
                            if not vif_data.empty:
                                vif_path = os.path.join(output_dir, final_vif_filename)
                                vif_data.to_csv(vif_path, index=False, encoding='utf-8-sig')
                                logger.info(f"Saved VIF scores to {vif_path}")
                            
                            # เลือกฟีเจอร์โดยพิจารณา multicollinearity
                            selected_features = select_top_features(corr_dat, X, n=10, config=config)
                            if selected_features:
                                model, metrics = train_model(scale_clean_engineer_dat, selected_features, scoring='roc_auc')
                                if model is not None and metrics:
                                    metrics_df = pd.DataFrame({
                                        'accuracy': [metrics.get('test_accuracy', np.nan)],
                                        'precision': [metrics.get('test_precision', np.nan)],
                                        'recall': [metrics.get('test_recall', np.nan)],
                                        'f1': [metrics.get('test_f1', np.nan)],
                                        'roc_auc': [metrics.get('test_roc_auc', np.nan)],
                                        'cross_validated_accuracy': [metrics.get('cross_validated_accuracy', np.nan)],
                                        'cross_validated_precision': [metrics.get('cross_validated_precision', np.nan)],
                                        'cross_validated_recall': [metrics.get('cross_validated_recall', np.nan)],
                                        'cross_validated_f1': [metrics.get('cross_validated_f1', np.nan)],
                                        'cross_validated_roc_auc': [metrics.get('cross_validated_roc_auc', np.nan)],
                                        'cv_score_mean': [metrics.get('cv_score_mean', np.nan)],
                                        'cv_score_std': [metrics.get('cv_score_std', np.nan)],
                                        'accuracy_ci_lower': [metrics.get('accuracy_ci_lower', np.nan)],
                                        'accuracy_ci_upper': [metrics.get('accuracy_ci_upper', np.nan)],
                                        'precision_ci_lower': [metrics.get('precision_ci_lower', np.nan)],
                                        'precision_ci_upper': [metrics.get('precision_ci_upper', np.nan)],
                                        'recall_ci_lower': [metrics.get('recall_ci_lower', np.nan)],
                                        'recall_ci_upper': [metrics.get('recall_ci_upper', np.nan)],
                                        'f1_ci_lower': [metrics.get('f1_ci_lower', np.nan)],
                                        'f1_ci_upper': [metrics.get('f1_ci_upper', np.nan)],
                                        'roc_auc_ci_lower': [metrics.get('roc_auc_ci_lower', np.nan)],
                                        'roc_auc_ci_upper': [metrics.get('roc_auc_ci_upper', np.nan)],
                                        'model_params': [metrics.get('model_params', {})],
                                        'oversampling_method': [metrics.get('oversampling_method', '')],
                                        'n_features_selected': [metrics.get('n_features_selected', 0)]
                                    })
                                    metrics_path = os.path.join(output_dir, final_metrics_filename)
                                    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
                                    logger.info(f"Saved model metrics to {metrics_path}")

                                    feature_importance_df = pd.DataFrame(metrics.get('feature_importance', {}))
                                    if not feature_importance_df.empty:
                                        feature_importance_path = os.path.join(output_dir, final_feature_importance_filename)
                                        feature_importance_df.to_csv(feature_importance_path, index=False, encoding='utf-8-sig')
                                        logger.info(f"Saved feature importance to {feature_importance_path}")

                                    model_path = os.path.join(model_dir, final_model_filename)
                                    dump(model, model_path)
                                    logger.info(f"Saved model to {model_path}")
                                else:
                                    logger.error("Failed to train random forest model.")
                            else:
                                logger.error("No features selected for training.")
                        else:
                            logger.error("Failed to compute correlations.")
                    else:
                        logger.error("Failed to preprocess data.")
                else:
                    logger.error("Failed to engineer features.")
            else:
                logger.error("Failed to transform data.")
        else:
            logger.error("Failed to clean data.")
    else:
        logger.error("Failed to load data.")