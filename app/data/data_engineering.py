
from typing import Optional
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from .data_cleaning import data_cleaning_fsk_v1, transform_data_fsk_v1, load_data_fsk_v1
import logging
import pandas as pd
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def features_engineer_fsk_v1(transformed_data: pd.DataFrame) -> Optional[pd.DataFrame]:
 
    try:
        if transformed_data is None or transformed_data.empty:
            logger.error("Input DataFrame is None or empty.")
            return None

        engineered_data = transformed_data.copy()
        logger.info(f"Columns in Transformed DataFrame: {engineered_data.columns.tolist()}")

        groups = {
            'spending': ['fht1', 'fht2'],
            'saving': ['fht3', 'fht4'],
            'paying_off': ['fht5', 'fht6'],
            'planning': ['fht7', 'fht8'],
            'debt': ['set9', 'set10'],
            'avoidance': ['kmsi1', 'kmsi2'],
            'worship': ['kmsi3', 'kmsi4'],
            'status': ['kmsi5', 'kmsi6'],
            'vigilance': ['kmsi7', 'kmsi8']
        }

        missing_cols = [col for group, cols in groups.items() for col in cols if col not in engineered_data.columns]
        if missing_cols:
            logger.error(f"Missing columns in DataFrame: {missing_cols}")
            return None

        for group, cols in groups.items():
            engineered_data[f'{group}_sum'] = engineered_data[cols].sum(axis=1)
            engineered_data[f'{group}_mean'] = engineered_data[cols].mean(axis=1)

        engineered_data['spending_to_saving_ratio'] = engineered_data['spending_sum'] / engineered_data['saving_sum'].replace(0, np.nan)
        engineered_data['debt_to_paying_off_ratio'] = engineered_data['debt_sum'] / engineered_data['paying_off_sum'].replace(0, np.nan)
        engineered_data['avoidance_to_vigilance_ratio'] = engineered_data['avoidance_sum'] / engineered_data['vigilance_sum'].replace(0, np.nan)
        engineered_data['worship_to_status_ratio'] = engineered_data['worship_sum'] / engineered_data['status_sum'].replace(0, np.nan)

        engineered_data['spending_avoidance_interaction'] = engineered_data['spending_mean'] * engineered_data['avoidance_mean']
        engineered_data['saving_planning_interaction'] = engineered_data['saving_mean'] * engineered_data['planning_mean']
        engineered_data['debt_vigilance_interaction'] = engineered_data['debt_mean'] * engineered_data['vigilance_mean']
        engineered_data['paying_off_worship_interaction'] = engineered_data['paying_off_mean'] * engineered_data['worship_mean']

        engineered_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        engineered_data.fillna(engineered_data.median(), inplace=True)

        return engineered_data

    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        return None
    
def handle_outliers(engineered_data: pd.DataFrame, outlier_method: str = 'median') -> Optional[pd.DataFrame]:
    
    try:
        if engineered_data is None or engineered_data.empty:
            logger.error("Input Engineered DataFrame is None or empty.")
            return None

        cleaned_data = engineered_data.copy()
        exclude_cols = ['ust']
        numeric_cols = [col for col in cleaned_data.columns if col not in exclude_cols and cleaned_data[col].dtype in [np.float64, np.int64]]
        
        for col in numeric_cols:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = cleaned_data[(cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)][col]
            if not outliers.empty:
                if outlier_method == 'median':
                    cleaned_data.loc[(cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound), col] = cleaned_data[col].median()
                elif outlier_method == 'cap':
                    cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
                elif outlier_method == 'remove':
                    initial_rows = len(cleaned_data)
                    cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
                    logger.info(f"Removed {initial_rows - len(cleaned_data)} rows due to outliers in {col}.")
                logger.info(f"Handled {len(outliers)} outliers in {col} using {outlier_method} method.")

        if cleaned_data.empty:
            logger.error("DataFrame is empty after outlier removal.")
            return None

        scaler = StandardScaler()
        cleaned_data[numeric_cols] = scaler.fit_transform(cleaned_data[numeric_cols])
        logger.info("Scaled numerical features using StandardScaler.")
        
        # Round all numeric columns (except 'ust') to 3 decimal places after scaling
        numeric_cols = cleaned_data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'ust']
        for col in numeric_cols:
            cleaned_data[col] = cleaned_data[col].round(3)
        logger.info(f"Rounded numeric columns {numeric_cols} to 3 decimal places after scaling")

        return cleaned_data

    except Exception as e:
        logger.error(f"Error during outlier removal: {str(e)}")
        return None

if __name__ == "__main__":
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Starting data processing pipeline...")
    raw_dat = load_data_fsk_v1()
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
            transformed_dat = transform_data_fsk_v1(cleaned_dat)
            logger.info("Data transformation completed.")
            if transformed_dat is not None:
                logger.info(f"Transformed DataFrame Shape: {transformed_dat.shape}")           
                logger.info(f"Transformed DataFrame Head:\n{transformed_dat.head().to_string()}")
                transformed_dat.to_csv(
                                        os.path.join(output_dir, f"transformed_dat_{timestamp}.csv"),
                                        index=False,
                                        encoding='utf-8-sig'
                ) 
                engineered_dat = features_engineer_fsk_v1(transformed_dat)
                logger.info("Feature engineering completed.")   
                if engineered_dat is not None:
                    logger.info(f"Engineered DataFrame Shape: {engineered_dat.shape}")
                    logger.info(f"Engineered DataFrame Head:\n{engineered_dat.head().to_string()}")
                    engineered_dat.to_csv(
                                            os.path.join(output_dir, f"engineered_dat_{timestamp}.csv"),
                                            index=False,
                                            encoding='utf-8-sig'
                    ) 
                    cleaned_engineered_dat = handle_outliers(engineered_dat)
                    logger.info("Outlier handling completed.")
                    if cleaned_engineered_dat is not None:
                        logger.info(f"Cleaned Engineered DataFrame Shape: {cleaned_engineered_dat.shape}")
                        logger.info(f"Cleaned Engineered DataFrame Head:\n{cleaned_engineered_dat.head().to_string()}")
                        cleaned_engineered_dat.to_csv(
                                                        os.path.join(output_dir, f"cleaned_engineered_dat_{timestamp}.csv"),
                                                        index=False,
                                                        encoding='utf-8-sig'
                        )                    
                        logger.info(f"Saved cleaned engineered data to {output_dir}/engineered_data_{timestamp}.csv")
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