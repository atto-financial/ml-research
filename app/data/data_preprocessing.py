import logging
import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, Dict, Union, Iterator
from sklearn.preprocessing import StandardScaler
from joblib import dump
from datetime import datetime
from scipy.stats import skew
from .data_loading import data_loading_fsk_v1
from .data_cleaning import data_cleaning_fsk_v1
from .data_transforming import data_transforming_fsk_v1
from .data_engineering import data_engineering_fsk_v1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def handle_outliers(df: pd.DataFrame, numeric_cols: list, outlier_method: str = 'median') -> pd.DataFrame:
    
    df_out = df.copy()
    for col in numeric_cols:
        col_skewness = skew(df_out[col].dropna())
        logger.info(f"Skewness of {col}: {col_skewness:.3f}")
        effective_method = outlier_method
        if abs(col_skewness) > 1:
            effective_method = 'median'
            logger.info(f"Column {col} is highly skewed (|skewness| > 1). Using 'median' method.")

        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        if outliers_count > 0:
            if effective_method == 'median':
                df_out.loc[outliers_mask, col] = df_out[col].median()
            elif effective_method == 'cap':
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
            elif effective_method == 'remove':
                df_out = df_out[~outliers_mask]
                logger.info(f"Removed {outliers_count} rows due to outliers in {col}.")
            logger.info(f"Handled {outliers_count} outliers in {col} using {effective_method} method.")
    return df_out

def data_preprocessing(
    engineer_dat: Union[pd.DataFrame, Iterator[pd.DataFrame]],
    outlier_method: str = 'median',
    metadata: Optional[Dict] = None
) -> Optional[Tuple[pd.DataFrame, StandardScaler]]:

    try:
        # Initialize result list for chunked data
        result_dfs = []
        is_chunked = isinstance(engineer_dat, Iterator)

        if not is_chunked:
            if engineer_dat is None or engineer_dat.empty:
                logger.error("Input DataFrame is None or empty.")
                return None, None
            engineer_dat = [engineer_dat]
        else:
            logger.info("Processing chunked input data.")

        scaler = StandardScaler()  # Initialize scaler (fit on combined data later if chunked)

        for chunk_idx, chunk in enumerate(engineer_dat):
            if chunk is None or chunk.empty:
                logger.warning(f"Chunk {chunk_idx} is None or empty. Skipping.")
                continue

            scale_clean_engineer_dat = chunk.copy()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Get numerical columns from metadata or infer
            exclude_cols = ['ust']
            if metadata and 'numerical_columns' in metadata:
                numeric_cols = metadata['numerical_columns']
            else:
                numeric_cols = [col for col in scale_clean_engineer_dat.columns if col not in exclude_cols and scale_clean_engineer_dat[col].dtype in [np.float64, np.int64]]

            if metadata and 'categorical_columns' in metadata:
                categorical_cols = metadata['categorical_columns']
                numeric_cols = [col for col in numeric_cols if col not in categorical_cols]
                logger.info(f"Excluded categorical columns from numeric_cols: {categorical_cols}")

            # Check zero-variance before processing
            zero_variance_cols = [col for col in numeric_cols if scale_clean_engineer_dat[col].var() == 0]
            if zero_variance_cols:
                logger.info(f"Found {len(zero_variance_cols)} features with zero variance before: {zero_variance_cols}")

            # Drop columns ending with '_sum' 
            columns_to_drop = [col for col in scale_clean_engineer_dat.columns if col.endswith('_sum')]
            if columns_to_drop:
                scale_clean_engineer_dat = scale_clean_engineer_dat.drop(columns=columns_to_drop)
                logger.info(f"Dropped columns: {columns_to_drop}")
                numeric_cols = [col for col in numeric_cols if col not in columns_to_drop]

            # Handle outliers on numerical columns
            scale_clean_engineer_dat = handle_outliers(scale_clean_engineer_dat, numeric_cols, outlier_method)

            if scale_clean_engineer_dat.empty:
                logger.error(f"Chunk {chunk_idx} is empty after outlier handling.")
                continue

            result_dfs.append(scale_clean_engineer_dat)
            logger.info(f"Processed chunk {chunk_idx} successfully.")

        if not result_dfs:
            logger.error("No valid chunks processed.")
            return None, None

        # Combine chunks
        combined_df = pd.concat(result_dfs, ignore_index=True)

        # Fit and transform scaler on combined numerical columns
        combined_df[numeric_cols] = scaler.fit_transform(combined_df[numeric_cols])
        logger.info("Scaled numerical features using StandardScaler on combined data.")

        # Check zero-variance after scaling
        zero_variance_cols_after = [col for col in numeric_cols if combined_df[col].var() == 0]
        if zero_variance_cols_after:
            combined_df = combined_df.drop(columns=zero_variance_cols_after)
            logger.info(f"Dropped {len(zero_variance_cols_after)} features with zero variance after scaling: {zero_variance_cols_after}")

        # Round numerical columns
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].round(3)
        logger.info("Rounded numeric columns to 3 decimal places after scaling.")

        return combined_df, scaler

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
    raw_dat, metadata = data_loading_fsk_v1()  # Assume returns (df, metadata)
    if raw_dat is not None:
        raw_dat.to_csv(os.path.join(output_dir, f"raw_dat_{timestamp}.csv"), index=False, encoding='utf-8-sig')
        cleaned_dat, metadata = data_cleaning_fsk_v1(raw_dat, outlier_method='median', metadata=metadata)
        if cleaned_dat is not None:
            cleaned_dat.to_csv(os.path.join(output_dir, f"cleaned_dat_{timestamp}.csv"), index=False, encoding='utf-8-sig')
            transform_dat = data_transforming_fsk_v1(cleaned_dat, metadata=metadata)
            if transform_dat is not None:
                transform_dat.to_csv(os.path.join(output_dir, f"transformed_dat_{timestamp}.csv"), index=False, encoding='utf-8-sig')
                engineer_dat = data_engineering_fsk_v1(transform_dat, metadata=metadata)
                if engineer_dat is not None:
                    engineer_dat.to_csv(os.path.join(output_dir, f"engineer_dat_{timestamp}.csv"), index=False, encoding='utf-8-sig')
                    scale_clean_engineer_dat, scaler = data_preprocessing(engineer_dat, outlier_method='median', metadata=metadata)
                    if scale_clean_engineer_dat is not None and scaler is not None:
                        scale_clean_engineer_dat.to_csv(os.path.join(output_dir, f"scale_clean_engineer_dat_{timestamp}.csv"), index=False, encoding='utf-8-sig')
                        logger.info(f"Saved preprocessed data to {output_dir}/scale_clean_engineer_dat_{timestamp}.csv")
                        
                        scaler_path = os.path.join(scaler_dir, f"{final_scaler_filename}.pkl")
                        dump(scaler, scaler_path)
                        logger.info(f"Saved scaler to {scaler_path}")
                        
                        logger.info("Data processing pipeline completed successfully.")
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