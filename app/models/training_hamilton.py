import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SMOTENC, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Tuple, List

logger = logging.getLogger(__name__)

def df_X_train(X_train_input: pd.DataFrame) -> pd.DataFrame:
    """Entry point for X_train."""
    return X_train_input

def s_y_train(y_train_input: pd.Series) -> pd.Series:
    """Entry point for y_train."""
    return y_train_input

def resampled_data(
    df_X_train: pd.DataFrame, 
    s_y_train: pd.Series, 
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply BorderlineSMOTE to handle class imbalance."""
    if df_X_train.empty or s_y_train.empty:
        return df_X_train, s_y_train
        
    k_neighbors = min(3, len(s_y_train) - 1) if len(s_y_train) > 1 else 1
    try:
        oversampler = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = oversampler.fit_resample(df_X_train, s_y_train)
        logger.info(f"Hamilton: Oversampling complete. Shape: {X_res.shape}")
        return X_res, y_res
    except Exception as e:
        logger.warning(f"Hamilton Oversampling failed: {e}. Returning original.")
        return df_X_train, s_y_train

def X_resampled(resampled_data: Tuple[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    return resampled_data[0]

def y_resampled(resampled_data: Tuple[pd.DataFrame, pd.Series]) -> pd.Series:
    return resampled_data[1]

def multicollinearity_free_features(
    X_resampled: pd.DataFrame, 
    y_resampled: pd.Series
) -> List[str]:
    """Iteratively remove multicollinear features based on VIF and RF importance."""
    current_features = X_resampled.columns.tolist()
    max_vif = 5.0
    
    if len(current_features) == 0:
        return []
        
    try:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_resampled, y_resampled)
        importances = model.feature_importances_
        imp_dict = dict(zip(current_features, importances))
    except Exception as e:
        logger.warning(f"Hamilton feature importance failed: {e}")
        imp_dict = {f: 1.0 for f in current_features}
    
    iteration = 0
    while True:
        X_numeric = X_resampled[current_features].copy()
        if X_numeric.empty:
            break
        X_numeric['intercept'] = 1
        
        try:
            if np.linalg.cond(X_numeric.values) > 1e10:
                X_numeric += np.random.normal(0, 1e-8, X_numeric.shape)
            vifs = [variance_inflation_factor(X_numeric.values, i) for i in range(len(current_features))]
        except Exception as e:
            logger.warning(f"Hamilton VIF computation failed on iteration {iteration}: {e}")
            break
            
        vif_df = pd.DataFrame({'feature': current_features, 'VIF': vifs})
        high_vif = vif_df[vif_df['VIF'] > max_vif]
        
        if high_vif.empty or iteration > len(current_features):
            break
            
        # Remove feature with worst importance
        worst_feature = min(high_vif['feature'].tolist(), key=lambda f: imp_dict.get(f, 0))
        current_features.remove(worst_feature)
        logger.info(f"Hamilton: Removed {worst_feature} due to high VIF.")
        iteration += 1
        
    return current_features
