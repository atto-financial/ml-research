from typing import Optional
from .data_loading import load_data_fsk_v1
from scipy.stats import skew
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def data_cleaning_fsk_v1(raw_dat: pd.DataFrame, outlier_method: str = 'median') -> Optional[pd.DataFrame]:
  
    try:
        if raw_dat is None or raw_dat.empty:
            logger.error("Input DataFrame is None or empty.")
            return None

        clean_dat = raw_dat.copy()
        initial_rows = len(clean_dat)

        categorical_columns = ['ust']
        numeric_columns = [
            'fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8',
            'set9', 'set10', 'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8'
        ]
        required_cols = categorical_columns + numeric_columns
        missing_cols = [col for col in required_cols if col not in clean_dat.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None

        clean_dat = clean_dat.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(clean_dat)} duplicate rows.")

        clean_dat[numeric_columns] = clean_dat[numeric_columns].apply(pd.to_numeric, errors='coerce')
        clean_dat['ust'] = pd.to_numeric(clean_dat['ust'], errors='coerce').astype('Int64')

        valid_values = [0, 1]
        invalid = clean_dat[~clean_dat['ust'].isin(valid_values)]['ust']
        if not invalid.empty:
            clean_dat = clean_dat[clean_dat['ust'].isin(valid_values)]
            logger.info(f"Dropped {len(invalid)} rows with invalid values in ust.")

        #Fill NaN using median for skewed distributions, mean for normal
        for col in numeric_columns:
            if clean_dat[col].isna().any():
                if clean_dat[col].dropna().count() < 3:
                    fill_value = clean_dat[col].median()
                    method = 'median (insufficient data)'
                else:
                    col_skew = skew(clean_dat[col].dropna())
                    skew_threshold = 1.0
                    fill_value = clean_dat[col].median() if abs(col_skew) > skew_threshold else clean_dat[col].mean()
                    method = 'median' if abs(col_skew) > skew_threshold else 'mean'
                clean_dat[col] = clean_dat[col].fillna(fill_value)
                logger.info(f"Filled NaN in {col} with {method} ({fill_value:.2f}), skewness: {col_skew:.2f}")
            else:
                logger.info(f"No NaN values in {col}")

        if clean_dat.empty:
            logger.error("DataFrame is empty after cleaning.")
            return None

        return clean_dat

    except Exception as e:
        logger.error(f"Error during cleaning: {str(e)}")
        return None


if __name__ == "__main__":
    raw_dat = load_data_fsk_v1()
    logger.info("Loading data completed.")
    if raw_dat is not None:
        logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
        logger.info(f"Raw DataFrame Head:\n{raw_dat.head().to_string()}")
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        logger.info("Data cleaning completed.")
        if cleaned_dat is not None:
            logger.info(f"Cleaned DataFrame Shape: {cleaned_dat.shape}")
            logger.info(f"Cleaned DataFrame Head:\n{cleaned_dat.head().to_string()}")
            logger.info(f"Summary Statistics:\n{cleaned_dat.describe().to_string()}")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")
        

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transform_data_fsk_v1(cleaned_data: pd.DataFrame) -> Optional[pd.DataFrame]:
   
    try:
        if cleaned_data is None or cleaned_data.empty:
            logger.error("Input DataFrame is None or empty.")
            return None

        transformed_data = cleaned_data.copy()

        columns_to_drop = ['user_id']
        existing_columns = [col for col in columns_to_drop if col in transformed_data.columns]
        if existing_columns:
            transformed_data = transformed_data.drop(columns=existing_columns)
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

        fht_columns = [col for col in fht_columns if col in transformed_data.columns]
        set_columns = [col for col in set_columns if col in transformed_data.columns]
        kmsi16_columns = [col for col in kmsi16_columns if col in transformed_data.columns]
        kmsi78_columns = [col for col in kmsi78_columns if col in transformed_data.columns]

        for col in fht_columns:
            invalid_rows = ~transformed_data[col].isin(fht_mapping.keys())
            if invalid_rows.any():
                transformed_data = transformed_data[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
            transformed_data[col] = transformed_data[col].map(fht_mapping)
            logger.info(f"Transformed values in {col} using fht_mapping.")

        for col in set_columns:
            invalid_rows = ~transformed_data[col].isin(set_mapping.keys())
            if invalid_rows.any():
                transformed_data = transformed_data[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
            transformed_data[col] = transformed_data[col].map(set_mapping)
            logger.info(f"Validated values in {col} using set_mapping.")

        for col in kmsi16_columns:
            invalid_rows = ~transformed_data[col].isin(kmsi16_mapping.keys())
            if invalid_rows.any():
                transformed_data = transformed_data[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
            transformed_data[col] = transformed_data[col].map(kmsi16_mapping)
            logger.info(f"Transformed values in {col} using kmsi16_mapping.")

        for col in kmsi78_columns:
            invalid_rows = ~transformed_data[col].isin(kmsi78_mapping.keys())
            if invalid_rows.any():
                transformed_data = transformed_data[~invalid_rows]
                logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
            transformed_data[col] = transformed_data[col].map(kmsi78_mapping)
            logger.info(f"Transformed values in {col} using kmsi78_mapping.")

        return transformed_data

    except Exception as e:
        logger.error(f"Error during transformation: {str(e)}")
        return None
       
if __name__ == "__main__":
    raw_dat = load_data_fsk_v1()
    logger.info("Loading data completed.")
    if raw_dat is not None:
        logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
        logger.info(f"Raw DataFrame Head:\n{raw_dat.head().to_string()}")
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        logger.info("Data cleaning completed.")
        if cleaned_dat is not None:
            logger.info(f"Cleaned DataFrame Shape: {cleaned_dat.shape}")
            logger.info(f"Cleaned DataFrame Head:\n{cleaned_dat.head().to_string()}")
            logger.info(f"Summary Statistics:\n{cleaned_dat.describe().to_string()}")
            transformed_data = transform_data_fsk_v1(cleaned_dat)
            logger.info("Data transformation completed.")
            if transformed_data is not None:
                logger.info(f"Transformed DataFrame Shape: {transformed_data.shape}")
                logger.info(f"Transformed DataFrame Head:\n{transformed_data.head().to_string()}")
                logger.info(f"Summary Statistics:\n{transformed_data.describe().to_string()}")
            else:
                print("Failed to transform data.")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")

    
