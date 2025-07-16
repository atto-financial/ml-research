import logging
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, Union, Iterator
from datetime import datetime
from .data_loading import data_loading_fsk_v1
from .data_cleaning import data_cleaning_fsk_v1
from .data_transforming import data_transforming_fsk_v1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_group_features(df: pd.DataFrame, group_name: str, columns: list) -> pd.DataFrame:
    
    if not columns:
        logger.info(f"No columns for group {group_name}.")
        return df

    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning(f"No valid columns for group {group_name} found in DataFrame.")
        return df

    df[f'{group_name}_score_sum'] = df[valid_columns].sum(axis=1)
    df[f'{group_name}_score_avg'] = df[valid_columns].mean(axis=1)
    df[f'{group_name}_score_var'] = df[valid_columns].var(axis=1).fillna(0)
    df[f'{group_name}_high_score_count'] = (df[valid_columns] == 3).sum(axis=1).astype('float64')
    logger.info(f"Created features for group {group_name}: sum, avg, var, high_score_count.")
    return df

def data_engineering_fsk_v1(
    transform_dat: Union[pd.DataFrame, Iterator[pd.DataFrame]], 
    metadata: Optional[Dict] = None
) -> Optional[pd.DataFrame]:
    
    try:
        # Initialize result list for chunked data
        result_dfs = []
        is_chunked = isinstance(transform_dat, Iterator)

        if not is_chunked:
            if transform_dat is None or transform_dat.empty:
                logger.error("Input DataFrame is None or empty.")
                return None
            transform_dat = [transform_dat]
        else:
            logger.info("Processing chunked input data.")

        for chunk_idx, chunk in enumerate(transform_dat):
            if chunk is None or chunk.empty:
                logger.warning(f"Chunk {chunk_idx} is None or empty. Skipping.")
                continue

            engineer_dat = chunk.copy()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Columns in chunk {chunk_idx}: {engineer_dat.columns.tolist()}")

            # Check zero-variance before engineering
            exclude_cols = ['ust']
            numeric_cols = [col for col in engineer_dat.columns 
                           if col not in exclude_cols and engineer_dat[col].dtype in [np.float64, np.int64]]
            zero_variance_before = [col for col in numeric_cols if engineer_dat[col].var() == 0]
            if zero_variance_before:
                logger.info(f"Found {len(zero_variance_before)} features with zero variance before: {zero_variance_before}")

            # Define groups (hard-coded)
            groups = {
                'spending': ['fht1', 'fht2'],
                'saving': ['fht3', 'fht4'],
                'payoff': ['fht5', 'fht6'],
                'planning': ['fht7', 'fht8'],
                'debt': ['set1', 'set2'],
                'loan': ['kmsi1', 'kmsi2'],
                'worship': ['kmsi3', 'kmsi4'],
                'extravagance': ['kmsi5', 'kmsi6'],
                'vigilance': ['kmsi7', 'kmsi8']
            }

            # Check for missing columns
            missing_cols = [col for group, cols in groups.items() for col in cols if col not in engineer_dat.columns]
            if missing_cols:
                logger.error(f"Missing columns in DataFrame: {missing_cols}")
                return None

            # Create group features
            for group, cols in groups.items():
                engineer_dat = create_group_features(engineer_dat, group, cols)

            # Create ratio features (hard-coded)
            ratios = [
                ('debt_to_payoff_ratio', 'debt_score_sum', 'payoff_score_sum'),
                ('loan_to_saving_ratio', 'loan_score_sum', 'saving_score_sum'),
                ('worship_to_vigilance_ratio', 'worship_score_sum', 'vigilance_score_sum'),
                ('extravagance_to_spending_ratio', 'extravagance_score_sum', 'spending_score_sum'),
                ('debt_to_saving_ratio', 'debt_score_sum', 'saving_score_sum'),
                ('worship_to_payoff_ratio', 'worship_score_sum', 'payoff_score_sum')
            ]
            for ratio_name, num_col, denom_col in ratios:
                engineer_dat[ratio_name] = engineer_dat[num_col] / (engineer_dat[denom_col] + 1)

            # Create interaction features (hard-coded)
            interactions = [
                ('debt_worship_interaction', 'debt_score_avg', 'worship_score_avg'),
                ('loan_extravagance_interaction', 'loan_score_avg', 'extravagance_score_avg'),
                ('payoff_planning_interaction', 'payoff_score_avg', 'planning_score_avg'),
                ('spending_vigilance_interaction', 'spending_score_avg', 'vigilance_score_avg'),
                ('debt_loan_interaction', 'debt_score_avg', 'loan_score_avg'),
                ('worship_extravagance_interaction', 'worship_score_avg', 'extravagance_score_avg')
            ]
            for inter_name, col1, col2 in interactions:
                engineer_dat[inter_name] = engineer_dat[col1] * engineer_dat[col2]

            # Handle categorical features from data_transforming.py 
            categorical_groups = {
                'category_group': [col for col in engineer_dat.columns if col.startswith('category_')],
                'type_group': [col for col in engineer_dat.columns if col.startswith('type_')],
                'region_group': [col for col in engineer_dat.columns if col.startswith('region_')]
            }
            if metadata and 'categorical_columns' in metadata:
                categorical_groups = {f"{col}_group": [c for c in engineer_dat.columns if c.startswith(f"{col}_")] 
                                     for col in metadata.get('categorical_columns', [])}

            for group, cols in categorical_groups.items():
                if cols:
                    engineer_dat[f'{group}_sum'] = engineer_dat[cols].sum(axis=1)
                    engineer_dat[f'{group}_count'] = (engineer_dat[cols] > 0).sum(axis=1)
                    logger.info(f"Created categorical features for {group}: sum, count.")

            # Handle NaN and Inf
            engineer_dat.replace([np.inf, -np.inf], np.nan, inplace=True)
            numeric_cols = engineer_dat.select_dtypes(include=[np.float64, np.int64]).columns
            if engineer_dat[numeric_cols].isna().sum().sum() > 0:
                logger.warning("Found NaN values after engineering. Filling with median.")
                engineer_dat[numeric_cols] = engineer_dat[numeric_cols].fillna(engineer_dat[numeric_cols].median())

            # Check zero-variance after engineering
            numeric_cols = [col for col in engineer_dat.columns 
                           if col not in exclude_cols and engineer_dat[col].dtype in [np.float64, np.int64]]
            zero_variance_after = [col for col in numeric_cols if engineer_dat[col].var() == 0]
            if zero_variance_after:
                logger.info(f"Found {len(zero_variance_after)} features with zero variance after: {zero_variance_after}")

            result_dfs.append(engineer_dat)
            logger.info(f"Processed chunk {chunk_idx} successfully.")

        if not result_dfs:
            logger.error("No valid chunks processed.")
            return None

        # Combine chunks
        final_df = pd.concat(result_dfs, ignore_index=True)
        logger.info("Feature engineering completed successfully.")
        return final_df

    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        return None

if __name__ == "__main__":
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Starting data processing pipeline...")

    raw_dat, metadata = data_loading_fsk_v1()  # Assume returns (df, metadata)
    if raw_dat is not None:
        raw_dat.to_csv(os.path.join(output_dir, f"raw_dat_{timestamp}.csv"), index=False, encoding='utf-8-sig')
        cleaned_dat, metadata = data_cleaning_fsk_v1(raw_dat, outlier_method='median', metadata=metadata)
        if cleaned_dat is not None:
            cleaned_dat.to_csv(os.path.join(output_dir, f"cleaned_dat_{timestamp}.csv"), index=False, encoding='utf-8-sig')
            transformed_dat = data_transforming_fsk_v1(cleaned_dat, metadata=metadata)
            if transformed_dat is not None:
                transformed_dat.to_csv(os.path.join(output_dir, f"transformed_dat_{timestamp}.csv"), index=False, encoding='utf-8-sig')
                engineer_dat = data_engineering_fsk_v1(transformed_dat, metadata=metadata)
                if engineer_dat is not None:
                    logger.info(f"Engineered DataFrame Shape: {engineer_dat.shape}")
                    logger.info(f"Engineered DataFrame Columns: {engineer_dat.columns.tolist()}")
                    logger.info(f"Engineered DataFrame Sample:\n{engineer_dat.sample(5).to_string()}")
                    engineer_dat.to_csv(os.path.join(output_dir, f"engineer_dat_{timestamp}.csv"), index=False, encoding='utf-8-sig')
                else:
                    logger.error("Failed to engineer features.")
            else:
                logger.error("Failed to transform data.")
        else:
            logger.error("Failed to clean data.")
    else:
        logger.error("Failed to load data.")