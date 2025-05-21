import pandas as pd
import logging
from typing import Optional
from .data_loading import data_loading_fsk_v1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
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
            'set1', 'set2', 'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8'
        ]
        required_cols = categorical_columns + numeric_columns
        missing_cols = [col for col in required_cols if col not in clean_dat.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        clean_dat = clean_dat.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(clean_dat)} duplicate rows.")

        for col in categorical_columns:
            clean_dat[col] = pd.to_numeric(clean_dat[col], errors='coerce').astype('Int64')
        
        for col in numeric_columns:
            clean_dat[col] = pd.to_numeric(clean_dat[col], errors='coerce').astype('Int64')

        valid_values = [0, 1]
        invalid = clean_dat[~clean_dat['ust'].isin(valid_values)]['ust']
        if not invalid.empty:
            clean_dat = clean_dat[clean_dat['ust'].isin(valid_values)]
            logger.info(f"Dropped {len(invalid)} rows with invalid values in ust.")

        if clean_dat.empty:
            logger.error("DataFrame is empty after cleaning.")
            return None

        return clean_dat

    except Exception as e:
        logger.error(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    raw_dat = data_loading_fsk_v1()
    logger.info("Loading data completed")
    if raw_dat is not None:
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        logger.info("Data cleaning completed")
        if cleaned_dat is not None:
            logger.info(f"Raw DataFrame Shape: {cleaned_dat.shape}")
            logger.info(f"Raw DataFrame Info:\n{cleaned_dat.info()}")
            logger.info(f"Raw DataFrame Sample:\n{cleaned_dat.sample(5).to_string()}")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")
        

    
