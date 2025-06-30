import logging
import pandas as pd
import numpy as np
import os
from typing import Optional
from datetime import datetime
from .data_loading import data_loading_fsk_v1
from .data_cleaning import data_cleaning_fsk_v1
from .data_transforming import data_transforming_fsk_v1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def data_engineering_fsk_v1(transform_dat: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        if transform_dat is None or transform_dat.empty:
            logger.error("Input DataFrame is None or empty.")
            return None

        engineer_dat = transform_dat.copy()
        logger.info(f"Columns in Transformed DataFrame: {engineer_dat.columns.tolist()}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exclude_cols = ['ust']
        numeric_cols = [col for col in engineer_dat.columns if col not in exclude_cols and engineer_dat[col].dtype in [np.float64, np.int64]]
        zero_variance_before = [col for col in numeric_cols if engineer_dat[col].var() == 0]
        if zero_variance_before:
            logger.info(f"Found {len(zero_variance_before)} features with zero variance before feature engineering: {zero_variance_before}")
        else:
            logger.info("No features with zero variance found before feature engineering.")

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
        
        missing_cols = [col for group, cols in groups.items() for col in cols if col not in engineer_dat.columns]
        if missing_cols:
            logger.error(f"Missing columns in DataFrame: {missing_cols}")
            return None

        for group, codes in groups.items():
            engineer_dat[f'{group}_score_sum'] = engineer_dat[codes].sum(axis=1)
            engineer_dat[f'{group}_score_avg'] = engineer_dat[codes].mean(axis=1)

        for group, codes in groups.items():
            
            # engineer_dat[f'{group}_high_score_count'] = (engineer_dat[codes] == 3).sum(axis=1).astype('float64')
            
            engineer_dat['debt_to_payoff_ratio'] = engineer_dat['debt_score_sum'] / (engineer_dat['payoff_score_sum'] + 1)
            engineer_dat['loan_to_saving_ratio'] = engineer_dat['loan_score_sum'] / (engineer_dat['saving_score_sum'] + 1)
            engineer_dat['worship_to_vigilance_ratio'] = engineer_dat['worship_score_sum'] / (engineer_dat['vigilance_score_sum'] + 1)
            engineer_dat['extravagance_to_spending_ratio'] = engineer_dat['extravagance_score_sum'] / (engineer_dat['spending_score_sum'] + 1)
            engineer_dat['debt_to_saving_ratio'] = engineer_dat['debt_score_sum'] / (engineer_dat['saving_score_sum'] + 1)
            engineer_dat['worship_to_payoff_ratio'] = engineer_dat['worship_score_sum'] / (engineer_dat['payoff_score_sum'] + 1)

            engineer_dat['debt_worship_interaction'] = engineer_dat['debt_score_avg'] * engineer_dat['worship_score_avg']
            engineer_dat['loan_extravagance_interaction'] = engineer_dat['loan_score_avg'] * engineer_dat['extravagance_score_avg']
            engineer_dat['payoff_planning_interaction'] = engineer_dat['payoff_score_avg'] * engineer_dat['planning_score_avg']
            engineer_dat['spending_vigilance_interaction'] = engineer_dat['spending_score_avg'] * engineer_dat['vigilance_score_avg']
            engineer_dat['debt_loan_interaction'] = engineer_dat['debt_score_avg'] * engineer_dat['loan_score_avg']
            engineer_dat['worship_extravagance_interaction'] = engineer_dat['worship_score_avg'] * engineer_dat['extravagance_score_avg']
            
            # engineer_dat[f'{group}_score_var'] = engineer_dat[codes].var(axis=1).fillna(0)

        engineer_dat.replace([np.inf, -np.inf], np.nan, inplace=True)
        engineer_dat.fillna(engineer_dat.median(), inplace=True)

        numeric_cols = [col for col in engineer_dat.columns if col not in exclude_cols and engineer_dat[col].dtype in [np.float64, np.int64]]
        
        zero_variance_after = [col for col in numeric_cols if engineer_dat[col].var() == 0]
        if zero_variance_after:
            logger.info(f"Found {len(zero_variance_after)} features with zero variance after feature engineering: {zero_variance_after}")
        else:
            logger.info("No features with zero variance found after feature engineering.")

        zero_variance_data = []
        for col in zero_variance_before:
            zero_variance_data.append({'feature': col, 'stage': 'Before'})
        for col in zero_variance_after:
            zero_variance_data.append({'feature': col, 'stage': 'After'})
        
        # if zero_variance_data:
        #     zero_variance_df = pd.DataFrame(zero_variance_data)
        #     zero_variance_path = os.path.join('output_data', f"zero_variance_features_{timestamp}.csv")
        #     zero_variance_df.to_csv(zero_variance_path, index=False, encoding='utf-8-sig')
        #     logger.info(f"Saved zero variance features to {zero_variance_path}")
        # else:
        #     logger.info("No zero variance features found before or after feature engineering")

        return engineer_dat

    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        return None

    
if __name__ == "__main__":
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Starting data processing pipeline...")
    raw_dat = data_loading_fsk_v1()
    logger.info("Loading data completed.")
    if raw_dat is not None:
        raw_dat.to_csv(
            os.path.join(output_dir, f"raw_dat_{timestamp}.csv"),
            index=False,
            encoding='utf-8-sig'
        )                            
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        logger.info("Data cleaning completed.")
        if cleaned_dat is not None:
            cleaned_dat.to_csv(
                os.path.join(output_dir, f"cleaned_dat_{timestamp}.csv"),
                index=False,
                encoding='utf-8-sig'
            )                             
            transformed_dat = data_transforming_fsk_v1(cleaned_dat)
            logger.info("Data transformation completed.")
            if transformed_dat is not None:
                transformed_dat.to_csv(
                    os.path.join(output_dir, f"transformed_dat_{timestamp}.csv"),
                    index=False,
                    encoding='utf-8-sig'
                ) 
                engineer_dat = data_engineering_fsk_v1(transformed_dat)
                logger.info("Feature engineering completed.")   
                if engineer_dat is not None:
                    logger.info(f"Engineered DataFrame Shape: {engineer_dat.shape}")
                    logger.info(f"Engineered DataFrame Info:\n{engineer_dat.info()}")
                    logger.info(f"Engineered DataFrame Sample:\n{engineer_dat.sample(5).to_string()}")
                    engineer_dat.to_csv(
                        os.path.join(output_dir, f"engineer_dat_{timestamp}.csv"),
                        index=False,
                        encoding='utf-8-sig'
                    ) 
                else:
                    print("Failed to engineer features.")
            else:
                print("Failed to transform data.")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")