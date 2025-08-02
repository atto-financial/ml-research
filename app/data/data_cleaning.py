import pandas as pd
import logging
from typing import Optional
from .data_loading import data_loading_fsk_v1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def data_cleaning_fsk_v1(raw_dat: pd.DataFrame, outlier_method: str = 'median') -> Optional[pd.DataFrame]:
   
    try:
        if raw_dat is None or raw_dat.empty:
            logger.error("Input DataFrame is None or empty.")
            return None

        clean_dat = raw_dat.copy()
        initial_rows = len(clean_dat)

        # Define column types
        target_column = 'ust'
        categorical_columns = []  
        # numerical_columns = [
        #     'fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8',
        #     'set1', 'set2', 'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8'
        # ]
        numerical_columns = [
            'fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8',
             'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8'
        ]
        required_cols = [target_column] + categorical_columns + numerical_columns + ['user_id']
        missing_cols = [col for col in required_cols if col not in clean_dat.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None

        # Remove duplicates
        clean_dat = clean_dat.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(clean_dat)} duplicate rows.")

        # Handle target column (ust)
        clean_dat[target_column] = clean_dat[target_column].astype('category')
        if clean_dat[target_column].isna().any():
            mode_value = clean_dat[target_column].mode()[0]
            clean_dat[target_column] = clean_dat[target_column].cat.add_categories([mode_value]).fillna(mode_value)
            logger.info(f"Imputed {clean_dat[target_column].isna().sum()} missing values in {target_column} with mode: {mode_value}")

        # Validate target values
        valid_values = [0, 1]
        invalid_rows = clean_dat[~clean_dat[target_column].astype(str).isin([str(v) for v in valid_values])]
        if not invalid_rows.empty:
            clean_dat = clean_dat[clean_dat[target_column].astype(str).isin([str(v) for v in valid_values])]
            logger.info(f"Dropped {len(invalid_rows)} rows with invalid values in {target_column}.")

        # Handle categorical columns (for future additions)
        for col in categorical_columns:
            clean_dat[col] = clean_dat[col].astype('category')
            if clean_dat[col].isna().any():
                mode_value = clean_dat[col].mode()[0]
                clean_dat[col] = clean_dat[col].cat.add_categories([mode_value]).fillna(mode_value)
                logger.info(f"Imputed {clean_dat[col].isna().sum()} missing values in {col} with mode: {mode_value}")

        # Handle numerical columns
        for col in numerical_columns:
            clean_dat[col] = pd.to_numeric(clean_dat[col], errors='coerce')
            if clean_dat[col].isna().any():
                if outlier_method == 'median':
                    fill_value = clean_dat[col].median()
                else:
                    fill_value = clean_dat[col].mean()
                clean_dat[col] = clean_dat[col].fillna(fill_value)
                logger.info(f"Imputed {clean_dat[col].isna().sum()} missing values in {col} with {outlier_method}: {fill_value}")
            clean_dat[col] = clean_dat[col].astype('float64')  

        if clean_dat.empty:
            logger.error("DataFrame is empty after cleaning.")
            return None

        logger.info(f"Cleaned DataFrame Shape: {clean_dat.shape}")
        return clean_dat

    except Exception as e:
        logger.error(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Starting data cleaning process")
    raw_dat = data_loading_fsk_v1()
    if raw_dat is not None:
        logger.info("Data loading completed")
        result = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        if result is not None:
            logger.info(f"Cleaned DataFrame Shape: {result.shape}")
            logger.info(f"Cleaned DataFrame Dtypes: {result.dtypes.to_dict()}")
            logger.info(f"Cleaned DataFrame Sample (5 rows):\n{result.sample(5).to_string()}")
        else:
            logger.error("Failed to clean data.")
    else:
        logger.error("Failed to load data.")