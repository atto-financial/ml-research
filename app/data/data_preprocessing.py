import logging
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from joblib import dump
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy.stats import skew
from pathlib import Path
from .data_loading import data_loading_fsk_v1
from .data_cleaning import data_cleaning_fsk_v1
from .data_transforming import data_transforming_fsk_v1
from .data_engineering import data_engineering_fsk_v1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def data_preprocessing(engineer_dat: pd.DataFrame, outlier_method: str = 'median') -> Optional[Tuple[pd.DataFrame, StandardScaler]]:
    try:
        if engineer_dat is None or engineer_dat.empty:
            logger.error("Input Engineered DataFrame is None or empty.")
            return None, None
            
        scale_clean_engineer_dat = engineer_dat.copy()
        exclude_cols = ['ust']
        numeric_cols = [col for col in scale_clean_engineer_dat.columns if col not in exclude_cols and scale_clean_engineer_dat[col].dtype in [np.float64, np.int64]]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zero_variance_cols = [col for col in numeric_cols if scale_clean_engineer_dat[col].var() == 0]
        if zero_variance_cols:
            # scale_clean_engineer_dat = scale_clean_engineer_dat.drop(columns=zero_variance_cols)
            # logger.info(f"Dropped {len(zero_variance_cols)} features with zero variance: {zero_variance_cols}")
            # zero_variance_df = pd.DataFrame(zero_variance_cols, columns=['zero_variance_feature'])
            # zero_variance_path = os.path.join('output_data', f"zero_variance_features_before_data_processing{timestamp}.csv")
            # zero_variance_df.to_csv(zero_variance_path, index=False, encoding='utf-8-sig')
            # logger.info(f"Saved zero variance features to {zero_variance_path}")
            numeric_cols = [col for col in scale_clean_engineer_dat.columns if col not in exclude_cols and scale_clean_engineer_dat[col].dtype in [np.float64, np.int64]]
        else:
            logger.info("No features with zero variance found before scaler.")

        if len(scale_clean_engineer_dat.columns) == 1 and 'ust' in scale_clean_engineer_dat.columns:
            logger.error("DataFrame contains only 'ust' column after dropping zero variance features.")
            return None, None

        columns_to_drop = [col for col in scale_clean_engineer_dat.columns if col.endswith('_sum')]
        if columns_to_drop:
            scale_clean_engineer_dat = scale_clean_engineer_dat.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns: {columns_to_drop}")
            numeric_cols = [col for col in scale_clean_engineer_dat.columns if col not in exclude_cols and scale_clean_engineer_dat[col].dtype in [np.float64, np.int64]]
        
        columns_to_drop = [col for col in scale_clean_engineer_dat.columns if col.startswith('debt_')]
        if columns_to_drop:
            scale_clean_engineer_dat = scale_clean_engineer_dat.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns: {columns_to_drop}")
            numeric_cols = [col for col in scale_clean_engineer_dat.columns if col not in exclude_cols and scale_clean_engineer_dat[col].dtype in [np.float64, np.int64]]
            
        for col in numeric_cols:
            col_skewness = skew(scale_clean_engineer_dat[col].dropna())
            logger.info(f"Skewness of {col}: {col_skewness:.3f}")
            effective_method = outlier_method
            if abs(col_skewness) > 1: 
                effective_method = 'median'
                logger.info(f"Column {col} is highly skewed (|skewness| > 1). Using 'median' method for outlier handling.")
            
            Q1 = scale_clean_engineer_dat[col].quantile(0.25)
            Q3 = scale_clean_engineer_dat[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = scale_clean_engineer_dat[(scale_clean_engineer_dat[col] < lower_bound) | (scale_clean_engineer_dat[col] > upper_bound)][col]
            if not outliers.empty:
                if effective_method == 'median':
                    scale_clean_engineer_dat.loc[(scale_clean_engineer_dat[col] < lower_bound) | (scale_clean_engineer_dat[col] > upper_bound), col] = scale_clean_engineer_dat[col].median()
                elif effective_method == 'cap':
                    scale_clean_engineer_dat[col] = scale_clean_engineer_dat[col].clip(lower=lower_bound, upper=upper_bound)
                elif effective_method == 'remove':
                    initial_rows = len(scale_clean_engineer_dat)
                    scale_clean_engineer_dat = scale_clean_engineer_dat[(scale_clean_engineer_dat[col] >= lower_bound) & (scale_clean_engineer_dat[col] <= upper_bound)]
                    logger.info(f"Removed {initial_rows - len(scale_clean_engineer_dat)} rows due to outliers in {col}.")
                logger.info(f"Handled {len(outliers)} outliers in {col} using {effective_method} method.")

        if scale_clean_engineer_dat.empty:
            logger.error("DataFrame is empty after outlier removal.")
            return None, None

        scaler = StandardScaler()
        scale_clean_engineer_dat[numeric_cols] = scaler.fit_transform(scale_clean_engineer_dat[numeric_cols])
        logger.info("Scaled numerical features using StandardScaler.")
        
        zero_variance_cols = [col for col in numeric_cols if scale_clean_engineer_dat[col].var() == 0]
        if zero_variance_cols:
            scale_clean_engineer_dat = scale_clean_engineer_dat.drop(columns=zero_variance_cols)
            logger.info(f"Dropped {len(zero_variance_cols)} features with zero variance: {zero_variance_cols}")
            numeric_cols = [col for col in scale_clean_engineer_dat.columns if col not in exclude_cols and scale_clean_engineer_dat[col].dtype in [np.float64, np.int64]]
        else:
            logger.info("No features with zero variance found after scaler.")
        
        numeric_cols = scale_clean_engineer_dat.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'ust']
        for col in numeric_cols:
            scale_clean_engineer_dat[col] = scale_clean_engineer_dat[col].round(3)
        logger.info(f"Rounded numeric columns to 3 decimal places after scaling")

        return scale_clean_engineer_dat, scaler

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return None, None

if __name__ == "__main__":
    output_dir = "output_data"
    scaler_dir = "save_scaler"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scaler_filename = None
    final_scaler_filename = scaler_filename if scaler_filename else f"custom_scaler_{timestamp}"

    logger.info("Starting data processing pipeline...")
    raw_dat = data_loading_fsk_v1()
    logger.info("Loading data completed.")
    if raw_dat is not None:
        logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
        logger.info(f"Raw DataFrame Head:\n{raw_dat.head().to_string()}")
        raw_dat.to_csv(
            os.path.join(output_dir, f"raw_dat_{timestamp}.csv"),
            index=False,
            encoding='utf-8-sig'
        )                            
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        logger.info("Data cleaning completed.")
        if cleaned_dat is not None:
            logger.info(f"Cleaned DataFrame Shape: {cleaned_dat.shape}")
            logger.info(f"Cleaned DataFrame Head:\n{cleaned_dat.head().to_string()}")
            logger.info(f"Cleaned DataFrame Info:\n{cleaned_dat.info()}")
            cleaned_dat.to_csv(
                os.path.join(output_dir, f"cleaned_dat_{timestamp}.csv"),
                index=False,
                encoding='utf-8-sig'
            )                             
            transform_dat = data_transforming_fsk_v1(cleaned_dat)
            logger.info("Data transformation completed.")
            if transform_dat is not None:
                logger.info(f"Transformed DataFrame Shape: {transform_dat.shape}")           
                logger.info(f"Transformed DataFrame Head:\n{transform_dat.head().to_string()}")
                transform_dat.to_csv(
                    os.path.join(output_dir, f"transformed_dat_{timestamp}.csv"),
                    index=False,
                    encoding='utf-8-sig'
                ) 
                engineer_dat = data_engineering_fsk_v1(transform_dat)
                logger.info("Feature engineering completed.")   
                if engineer_dat is not None:
                    logger.info(f"Engineered DataFrame Shape: {engineer_dat.shape}")
                    logger.info(f"Engineered DataFrame Head:\n{engineer_dat.head().to_string()}")
                    engineer_dat.to_csv(
                        os.path.join(output_dir, f"engineer_dat_{timestamp}.csv"),
                        index=False,
                        encoding='utf-8-sig'
                    ) 
                    scale_clean_engineer_dat, scaler = data_preprocessing(engineer_dat)
                    logger.info("Outlier handling completed.")
                    if scale_clean_engineer_dat is not None and scaler is not None:
                        logger.info(f"Cleaned Engineered DataFrame Shape: {scale_clean_engineer_dat.shape}")
                        logger.info(f"Cleaned Engineered DataFrame Head:\n{scale_clean_engineer_dat.head().to_string()}")
                        scale_clean_engineer_dat.to_csv(
                            os.path.join(output_dir, f"scale_clean_engineer_dat_{timestamp}.csv"),
                            index=False,
                            encoding='utf-8-sig'
                        )                    
                        logger.info(f"Saved cleaned engineered data to {output_dir}/scale_clean_engineer_dat_{timestamp}.csv")
                        
                        scaler_path = os.path.join(scaler_dir, f"{final_scaler_filename}.pkl")
                        dump(scaler, scaler_path)
                        logger.info(f"Saved scaler to {scaler_path}")
                        
                        logger.info("Data processing pipeline completed successfully.")
                    else:
                        print("Failed to remove outliers or scale data.")
                else:
                    print("Failed to engineer features.")
            else:
                print("Failed to transform data.")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")