from typing import Optional
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from .data_loading import data_loading_fsk_v1
from .data_cleaning import data_cleaning_fsk_v1
from .data_transforming import data_transform_fsk_v1
from .data_engineering import data_engineer_fsk_v1
import logging
import pandas as pd
import numpy as np
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def data_outliers(engineer_dat: pd.DataFrame, outlier_method: str = 'median') -> Optional[pd.DataFrame]:
    
    try:
        if engineer_dat is None or engineer_dat.empty:
            logger.error("Input Engineered DataFrame is None or empty.")
            return None
            
        clean_engineer_dat = engineer_dat.copy()
        exclude_cols = ['ust']
        numeric_cols = [col for col in clean_engineer_dat.columns if col not in exclude_cols and clean_engineer_dat[col].dtype in [np.float64, np.int64]]
        
        columns_to_drop = [col for col in clean_engineer_dat.columns if col.endswith('_sum')]
        if columns_to_drop:
            clean_engineer_dat = clean_engineer_dat.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns: {columns_to_drop}")
            numeric_cols = [col for col in clean_engineer_dat.columns if col not in exclude_cols and clean_engineer_dat[col].dtype in [np.float64, np.int64]]
            
        for col in numeric_cols:
            Q1 = clean_engineer_dat[col].quantile(0.25)
            Q3 = clean_engineer_dat[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = clean_engineer_dat[(clean_engineer_dat[col] < lower_bound) | (clean_engineer_dat[col] > upper_bound)][col]
            if not outliers.empty:
                if outlier_method == 'median':
                    clean_engineer_dat.loc[(clean_engineer_dat[col] < lower_bound) | (clean_engineer_dat[col] > upper_bound), col] = clean_engineer_dat[col].median()
                elif outlier_method == 'cap':
                    clean_engineer_dat[col] = clean_engineer_dat[col].clip(lower=lower_bound, upper=upper_bound)
                elif outlier_method == 'remove':
                    initial_rows = len(clean_engineer_dat)
                    clean_engineer_dat = clean_engineer_dat[(clean_engineer_dat[col] >= lower_bound) & (clean_engineer_dat[col] <= upper_bound)]
                    logger.info(f"Removed {initial_rows - len(clean_engineer_dat)} rows due to outliers in {col}.")
                logger.info(f"Handled {len(outliers)} outliers in {col} using {outlier_method} method.")

        if clean_engineer_dat.empty:
            logger.error("DataFrame is empty after outlier removal.")
            return None

        scaler = StandardScaler()
        clean_engineer_dat[numeric_cols] = scaler.fit_transform(clean_engineer_dat[numeric_cols])
        logger.info("Scaled numerical features using StandardScaler.")
        
        numeric_cols = clean_engineer_dat.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'ust']
        for col in numeric_cols:
            clean_engineer_dat[col] = clean_engineer_dat[col].round(3)
        logger.info(f"Rounded numeric columns {numeric_cols} to 3 decimal places after scaling")

        return clean_engineer_dat

    except Exception as e:
        logger.error(f"Error during outlier removal: {str(e)}")
        return None

if __name__ == "__main__":
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            transform_dat = data_transform_fsk_v1(cleaned_dat)
            logger.info("Data transformation completed.")
            if transform_dat is not None:
                logger.info(f"Transformed DataFrame Shape: {transform_dat.shape}")           
                logger.info(f"Transformed DataFrame Head:\n{transform_dat.head().to_string()}")
                transform_dat.to_csv(
                                        os.path.join(output_dir, f"transformed_dat_{timestamp}.csv"),
                                        index=False,
                                        encoding='utf-8-sig'
                ) 
                engineer_dat = data_engineer_fsk_v1(transform_dat)
                logger.info("Feature engineering completed.")   
                if engineer_dat is not None:
                    logger.info(f"Engineered DataFrame Shape: {engineer_dat.shape}")
                    logger.info(f"Engineered DataFrame Head:\n{engineer_dat.head().to_string()}")
                    engineer_dat.to_csv(
                                            os.path.join(output_dir, f"engineer_datt_{timestamp}.csv"),
                                            index=False,
                                            encoding='utf-8-sig'
                    ) 
                    clean_engineer_dat = data_outliers(engineer_dat)
                    logger.info("Outlier handling completed.")
                    if clean_engineer_dat is not None:
                        logger.info(f"Cleaned Engineered DataFrame Shape: {clean_engineer_dat.shape}")
                        logger.info(f"Cleaned Engineered DataFrame Head:\n{clean_engineer_dat.head().to_string()}")
                        clean_engineer_dat.to_csv(
                                                        os.path.join(output_dir, f"clean_engineer_dat_{timestamp}.csv"),
                                                        index=False,
                                                        encoding='utf-8-sig'
                        )                    
                        logger.info(f"Saved cleane engineer data to {output_dir}/engineer_dat_{timestamp}.csv")
                        logger.info("Data processing pipeline completed successfully.")
                    else:
                        print("Failed to remove outliers.")
                else:
                    print("Failed to engineer features.")
            else:
                print("Failed to transform data.")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")