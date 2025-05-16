from typing import Optional
from .data_loading import data_loading_fsk_v1
from .data_cleaning import data_cleaning_fsk_v1
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def data_transform_fsk_v1(clean_dat: pd.DataFrame) -> Optional[pd.DataFrame]:
   
    try:
        if clean_dat is None or clean_dat.empty:
            logger.error("Input DataFrame is None or empty.")
            return None

        transform_dat = clean_dat.copy()
        columns_to_drop = ['user_id']
        existing_columns = [col for col in columns_to_drop if col in transform_dat.columns]
        
        if existing_columns:
            transform_dat = clean_dat.drop(columns=existing_columns)
            logger.info(f"Dropped columns: {existing_columns}")
        else:
            logger.info("No columns to drop found in input data.")

        fht_mapping = {1: 3, 2: 2, 3: 1}
        set_mapping = {1: 1, 2: 2, 3: 3}
        kmsi16_mapping = {1: 1, 2: 3}
        kmsi78_mapping = {1: 3, 2: 1}

        fht_columns = ['fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8']
        set_columns = ['set9', 'set10']
        kmsi16_columns = ['kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6']
        kmsi78_columns = ['kmsi7', 'kmsi8']

        fht_columns = [col for col in fht_columns if col in transform_dat.columns]
        set_columns = [col for col in set_columns if col in transform_dat.columns]
        kmsi16_columns = [col for col in kmsi16_columns if col in transform_dat.columns]
        kmsi78_columns = [col for col in kmsi78_columns if col in transform_dat.columns]

        for col in fht_columns:
            invalid_rows = ~transform_dat[col].isin(fht_mapping.keys())
            if invalid_rows.any():
                transform_dat = transform_dat[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
            transform_dat[col] = transform_dat[col].map(fht_mapping)
            logger.info(f"Transformed values in {col} using fht_mapping.")

        for col in set_columns:
            invalid_rows = ~transform_dat[col].isin(set_mapping.keys())
            if invalid_rows.any():
                transform_dat = transform_dat[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
            transform_dat[col] = transform_dat[col].map(set_mapping)
            logger.info(f"Validated values in {col} using set_mapping.")

        for col in kmsi16_columns:
            invalid_rows = ~transform_dat[col].isin(kmsi16_mapping.keys())
            if invalid_rows.any():
                transform_dat = transform_dat[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
            transform_dat[col] = transform_dat[col].map(kmsi16_mapping)
            logger.info(f"Transformed values in {col} using kmsi16_mapping.")

        for col in kmsi78_columns:
            invalid_rows = ~transform_dat[col].isin(kmsi78_mapping.keys())
            if invalid_rows.any():
                transform_dat = transform_dat[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
            transform_dat[col] = transform_dat[col].map(kmsi78_mapping)
            logger.info(f"Transformed values in {col} using kmsi78_mapping.")

        return transform_dat

    except Exception as e:
        logger.error(f"Error during transformation: {str(e)}")
        return None
       
if __name__ == "__main__":
    raw_dat = data_loading_fsk_v1()
    logger.info("Loading data completed.")
    if raw_dat is not None:
        clean_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        logger.info("Data cleaning completed.")
        if clean_dat is not None:
            transform_dat = data_transform_fsk_v1(clean_dat)
            logger.info("Data transformation completed.")
            if transform_dat is not None:
                logger.info(f"Raw DataFrame Shape: {transform_dat.shape}")
                logger.info(f"Raw DataFrame Columns: {transform_dat.columns.tolist()}")
                logger.info(f"Raw DataFrame Info:\n{transform_dat.info()}")
                logger.info(f"Raw DataFrame Sample:\n{transform_dat.sample(5).to_string()}")
            else:
                print("Failed to transform data.")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")

