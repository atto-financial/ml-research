import pandas as pd
import numpy as np
import logging
import os
from typing import Optional
from datetime import datetime
from .data_loading import data_loading_fsk_v1
from .data_cleaning import data_cleaning_fsk_v1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def data_transforming_fsk_v1(clean_dat: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        if clean_dat is None or clean_dat.empty:
            logger.error("Input DataFrame is None or empty.")
            return None

        transform_dat = clean_dat.copy()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exclude_cols = ['ust']
        numeric_cols = [col for col in transform_dat.columns if col not in exclude_cols and transform_dat[col].dtype in [np.float64, np.int64]]
        zero_variance_cols = [col for col in numeric_cols if transform_dat[col].var() == 0]
        if zero_variance_cols:
            logger.info(f"Found {len(zero_variance_cols)} features with zero variance before mapping: {zero_variance_cols}")
            # zero_variance_df = pd.DataFrame(zero_variance_cols, columns=['zero_variance_feature'])
            # zero_variance_path = os.path.join('output_data', f"zero_variance_features_before_mapping_{timestamp}.csv")
            # zero_variance_df.to_csv(zero_variance_path, index=False, encoding='utf-8-sig')
            # logger.info(f"Saved zero variance features to {zero_variance_path}")
        else:
            logger.info("No features with zero variance found before mapping.")

        columns_to_drop = ['user_id']
        existing_columns = [col for col in columns_to_drop if col in transform_dat.columns]
        
        if existing_columns:
            transform_dat = transform_dat.drop(columns=existing_columns)
            logger.info(f"Dropped columns: {existing_columns}")
        else:
            logger.info("No columns to drop found in input data.")

        fht_mapping = {1: 3, 2: 2, 3: 1}
        set_mapping = {1: 1, 2: 2, 3: 3}
        # kmsi16_mapping = {1: 1, 2: 3}
        # kmsi78_mapping = {1: 3, 2: 1}
        kmsi16_mapping = {1: 1,2: 2, 3: 3}
        kmsi78_mapping = {1: 3, 2: 2, 3: 1}

        fht_columns = ['fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8']
        set_columns = ['set1', 'set2']
        kmsi16_columns = ['kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6']
        kmsi78_columns = ['kmsi7', 'kmsi8']

        fht_columns = [col for col in fht_columns if col in transform_dat.columns]
        set_columns = [col for col in set_columns if col in transform_dat.columns]
        kmsi16_columns = [col for col in kmsi16_columns if col in transform_dat.columns]
        kmsi78_columns = [col for col in kmsi78_columns if col in transform_dat.columns]

        mapped_columns = fht_columns + set_columns + kmsi16_columns + kmsi78_columns

        for col in fht_columns:
            invalid_rows = ~transform_dat[col].isin(fht_mapping.keys())
            if invalid_rows.any():
                transform_dat = transform_dat[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")

        for col in set_columns:
            invalid_rows = ~transform_dat[col].isin(set_mapping.keys())
            if invalid_rows.any():
                transform_dat = transform_dat[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")

        for col in kmsi16_columns:
            invalid_rows = ~transform_dat[col].isin(kmsi16_mapping.keys())
            if invalid_rows.any():
                transform_dat = transform_dat[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")

        for col in kmsi78_columns:
            invalid_rows = ~transform_dat[col].isin(kmsi78_mapping.keys())
            if invalid_rows.any():
                transform_dat = transform_dat[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
                
        if fht_columns:
            transform_dat[fht_columns] = transform_dat[fht_columns].apply(
                lambda x: x.map(fht_mapping)
            ).astype(np.float64)
            logger.info(f"Transformed values in {fht_columns} using fht_mapping to float64.")

        if set_columns:
            transform_dat[set_columns] = transform_dat[set_columns].apply(
                lambda x: x.map(set_mapping)
            ).astype(np.float64)
            logger.info(f"Transformed values in {set_columns} using set_mapping to float64.")

        if kmsi16_columns:
            transform_dat[kmsi16_columns] = transform_dat[kmsi16_columns].apply(
                lambda x: x.map(kmsi16_mapping)
            ).astype(np.float64)
            logger.info(f"Transformed values in {kmsi16_columns} using kmsi16_mapping to float64.")

        if kmsi78_columns:
            transform_dat[kmsi78_columns] = transform_dat[kmsi78_columns].apply(
                lambda x: x.map(kmsi78_mapping)
            ).astype(np.float64)
            logger.info(f"Transformed values in {kmsi78_columns} using kmsi78_mapping to float64.")
            
        if mapped_columns:
            nan_count = transform_dat[mapped_columns].isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values after mapping. Filling with column medians.")
                transform_dat[mapped_columns] = transform_dat[mapped_columns].fillna(
                    transform_dat[mapped_columns].median()
                )

        if 'ust' in transform_dat.columns:
            invalid_ust = ~transform_dat['ust'].isin([0, 1])
            if invalid_ust.any():
                logger.warning(f"Dropped {invalid_ust.sum()} rows with invalid values in ust")
                transform_dat = transform_dat[~invalid_ust]
            
            transform_dat['ust'] = transform_dat['ust'].astype(np.int64)
            logger.info("Encoded ust column to int64.")
            
        logger.info("Data transformation completed successfully.")
        
        return transform_dat

    except Exception as e:
        logger.error(f"Error during transformation: {str(e)}")
        return None
       
if __name__ == "__main__":
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)  
    raw_dat = data_loading_fsk_v1()
    logger.info("Loading data completed.")
    if raw_dat is not None:
        clean_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        logger.info("Data cleaning completed.")
        if clean_dat is not None:
            transform_dat = data_transforming_fsk_v1(clean_dat)
            logger.info("Data transformation completed.")
            if transform_dat is not None:
                logger.info(f"Transformed DataFrame Shape: {transform_dat.shape}")
                logger.info(f"Transformed DataFrame Columns: {transform_dat.columns.tolist()}")
                logger.info(f"Transformed DataFrame Info:\n{transform_dat.info()}")
                logger.info(f"Transformed DataFrame Sample:\n{transform_dat.sample(5).to_string()}")
            else:
                print("Failed to transform data.")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")