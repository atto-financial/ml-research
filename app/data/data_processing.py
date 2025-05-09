import pandas as pd
import numpy as np
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_cleaning(raw_dat: pd.DataFrame) -> Optional[pd.DataFrame]:

    try:
        df = raw_dat.copy()
        logger.info(f"Starting data cleaning for DataFrame with {len(df)} rows and {len(df.columns)} columns.")

        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows.")

        numeric_columns = [
            'fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8',
            'cdd9', 'cdd10', 'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8',
            'ins'
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') 
        df['ust'] = df['ust'].astype(int)  

        for col in numeric_columns:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {df[col].isna().sum()} missing values in {col} with median {median_val}.")

        if df['ust'].isna().sum() > 0:
            df = df.dropna(subset=['ust'])
            logger.info(f"Dropped rows with missing 'ust' values.")

        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not outliers.empty:
                df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = df[col].median()
                logger.info(f"Replaced {len(outliers)} outliers in {col} with median.")

        if df.empty:
            logger.error("DataFrame is empty after cleaning.")
            return None

        logger.info(f"Data cleaning completed. Final DataFrame has {len(df)} rows.")
        return df

    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        return None
    
