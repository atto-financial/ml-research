from typing import Optional
from datetime import datetime
from app.data.data_engineering import features_engineer_fsk_v1, handle_outliers
from app.data.data_cleaning import data_cleaning_fsk_v1, transform_data_fsk_v1, load_data_fsk_v1
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_correlations(engineered_data: pd.DataFrame) -> Optional[pd.DataFrame]:
   
    try:
        if engineered_data is None or engineered_data.empty:
            logger.error("Input DataFrame is None or empty.")
            return None

        exclude_cols = ['ust'] + [f'{group}_sum' for group in ['spending', 'saving', 'paying_off', 'planning', 'debt', 'avoidance', 'worship', 'status', 'vigilance']]
        numeric_cols = [col for col in engineered_data.columns if col not in exclude_cols and engineered_data[col].dtype in [np.float64, np.int64]]

        correlations = {}
        for col in numeric_cols:
            if engineered_data[col].std() == 0:
                logger.warning(f"Skipping {col}: Variance is zero (all values are the same).")
                correlations[col] = {'Pearson': np.nan, 'Spearman': np.nan}
                continue
            pearson_corr = engineered_data[col].corr(engineered_data['ust'], method='pearson')
            spearman_corr = engineered_data[col].corr(engineered_data['ust'], method='spearman')
            correlations[col] = {'Pearson': pearson_corr, 'Spearman': spearman_corr}

        corr_df = pd.DataFrame.from_dict(correlations, orient='index')
        corr_df = corr_df.sort_values(by='Pearson', ascending=False)

        try:
            corr_df.to_csv(
                os.path.join('output_data', f"correlations_with_ust_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                index=True,
                encoding='utf-8-sig'
            )
            logger.info("Saved correlations to 'correlations_with_ust.csv'.")
        except Exception as e:
            logger.error(f"Failed to save correlations: {e}")

        return corr_df

    except Exception as e:
        logger.error(f"Error during correlation computation: {str(e)}")
        return None

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_correlations(corr_df: pd.DataFrame, output_dir: str = 'plots') -> None:
    if corr_df is None or corr_df.empty:
        logger.error("Correlation DataFrame is None or empty.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return

    corr_df = corr_df.fillna(0)
    logger.info(f"Correlation DataFrame:\n{corr_df.to_string()}")

    
    if (corr_df['Pearson'] >= 0).all():
        logger.warning("All Pearson correlations are non-negative. Bar plot may show only positive side.")
        top_positive_pearson = corr_df.head(10)
        top_negative_pearson = pd.DataFrame(columns=corr_df.columns) 
    elif (corr_df['Pearson'] <= 0).all():
        logger.warning("All Pearson correlations are non-positive. Bar plot may show only negative side.")
        top_positive_pearson = pd.DataFrame(columns=corr_df.columns)  
        top_negative_pearson = corr_df.tail(10)
    else:
        top_positive_pearson = corr_df[corr_df['Pearson'] > 0].head(10)
        top_negative_pearson = corr_df[corr_df['Pearson'] < 0].tail(10)

    top_corr = pd.concat([top_positive_pearson, top_negative_pearson])
    if top_corr.empty:
        logger.warning("No features to plot in top_corr.")
        return

    plt.figure(figsize=(10, len(top_corr) * 0.5))  
    sns.barplot(x=top_corr['Pearson'], y=top_corr.index)
    plt.title('Top 10 Positive and Negative Pearson Correlations with ust')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pearson_correlations_with_ust.png')
    plt.close()
    logger.info(f"Saved Pearson correlation plot to {output_dir}/pearson_correlations_with_ust.png")

    plt.figure(figsize=(10, len(top_corr) * 0.5))
    sns.barplot(x=top_corr['Spearman'], y=top_corr.index)
    plt.title('Top 10 Positive and Negative Spearman Correlations with ust')
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spearman_correlations_with_ust.png')
    plt.close()
    logger.info(f"Saved Spearman correlation plot to {output_dir}/spearman_correlations_with_ust.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=corr_df['Pearson'], y=corr_df['Spearman'])
    plt.title('Pearson vs Spearman Correlations with ust')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Spearman Correlation')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    for i, feature in enumerate(corr_df.index):
        if abs(corr_df['Pearson'][i] - corr_df['Spearman'][i]) > 0.1:
            plt.text(corr_df['Pearson'][i], corr_df['Spearman'][i], feature, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pearson_vs_spearman.png')
    plt.close()
    logger.info(f"Saved Pearson vs Spearman plot to {output_dir}/pearson_vs_spearman.png")

if __name__ == "__main__":
    output_dir = "output_data"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        exit(1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dat = load_data_fsk_v1()
    if raw_dat is not None:
        logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        if cleaned_dat is not None:
            logger.info(f"Cleaned DataFrame Shape: {cleaned_dat.shape}")
            transformed_dat = transform_data_fsk_v1(cleaned_dat)
            if transformed_dat is not None:
                logger.info(f"Transformed DataFrame Shape: {transformed_dat.shape}")
                engineered_data = features_engineer_fsk_v1(transformed_dat)
                if engineered_data is not None:
                    logger.info(f"Features Engineered DataFrame Shape: {engineered_data.shape}")
                    logger.info(f"Columns in Features Engineered DataFrame: {engineered_data.columns.tolist()}")
                    cleaned_engineered_dat = handle_outliers(engineered_data)
                    if cleaned_engineered_dat is not None:
                        logger.info(f"Cleaned Engineered DataFrame Shape (After Outlier Removal): {cleaned_engineered_dat.shape}")
                        cleaned_engineered_dat.to_csv(
                                os.path.join(output_dir, f"cleaned_engineered_data_{timestamp}.csv"),
                                index=False,
                                encoding='utf-8-sig'
                            )
                        logger.info(f"Saved cleaned engineered data to {output_dir}/cleaned_engineered_data_{timestamp}.csv")
                        corr_df = compute_correlations(cleaned_engineered_dat)
                        if corr_df is not None:
                            logger.info("Correlations with ust (Pearson and Spearman):")
                            logger.info(f"\n{corr_df}")
                            visualize_correlations(corr_df)
                            logger.info("Correlation visualizations saved in 'plots' directory.")
                        else:
                            logger.error("Failed to compute correlations.")
                    else:
                        logger.error("Failed to remove outliers.")
                else:
                    logger.error("Failed to engineer features.")
            else:
                logger.error("Failed to transform data.")
        else:
            logger.error("Failed to clean data.")
    else:
        logger.error("Failed to load data.")