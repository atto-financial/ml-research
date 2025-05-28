from typing import Optional
from datetime import datetime
from app.data.data_loading import data_loading_fsk_v1
from app.data.data_cleaning import data_cleaning_fsk_v1
from app.data.data_transforming import data_transforming_fsk_v1
from app.data.data_engineering import data_engineering_fsk_v1
from app.data.data_preprocessing import data_preprocessing
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compute_correlations(scale_clean_engineer_dat: pd.DataFrame) -> Optional[pd.DataFrame]:
   
    try:
        if scale_clean_engineer_dat is None or scale_clean_engineer_dat.empty:
            logger.error("Input DataFrame is None or empty.")
            return None

        exclude_cols = ['ust'] + [f'{group}_sum' for group in ['spending', 'saving', 'paying_off', 'planning', 'debt', 'avoidance', 'worship', 'status', 'vigilance']]
        numeric_cols = [col for col in scale_clean_engineer_dat.columns if col not in exclude_cols and scale_clean_engineer_dat[col].dtype in [np.float64, np.int64]]

        correlations = {}
        for col in numeric_cols:
            if scale_clean_engineer_dat[col].std() == 0:
                logger.warning(f"Skipping {col}: Variance is zero (all values are the same).")
                correlations[col] = {'Pearson': np.nan, 'Spearman': np.nan}
                continue
            pearson_corr = scale_clean_engineer_dat[col].corr(scale_clean_engineer_dat['ust'], method='pearson')
            spearman_corr = scale_clean_engineer_dat[col].corr(scale_clean_engineer_dat['ust'], method='spearman')
            correlations[col] = {'Pearson': pearson_corr, 'Spearman': spearman_corr}

        corr_dat = pd.DataFrame.from_dict(correlations, orient='index')
        corr_dat = corr_dat.sort_values(by='Pearson', ascending=False)

        return corr_dat

    except Exception as e:
        logger.error(f"Error during correlation computation: {str(e)}")
        return None


def visualize_correlations(corr_dat: pd.DataFrame, output_dir: str = 'plots', plot_prefix: str = None) -> None:
    if corr_dat is None or corr_dat.empty:
        raise ValueError("Correlation DataFrame is None or empty.")

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise Exception(f"Failed to create output directory {output_dir}: {e}")

    corr_dat = corr_dat.fillna(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pearson_plot_filename = f"{plot_prefix}_pearson_correlations_{timestamp}.png" if plot_prefix else f"pearson_correlations_{timestamp}.png"
    spearman_plot_filename = f"{plot_prefix}_spearman_correlations_{timestamp}.png" if plot_prefix else f"spearman_correlations_{timestamp}.png"
    scatter_plot_filename = f"{plot_prefix}_pearson_vs_spearman_{timestamp}.png" if plot_prefix else f"pearson_vs_spearman_{timestamp}.png"

    if (corr_dat['Pearson'] >= 0).all():
        top_positive_pearson = corr_dat.head(10)
        top_negative_pearson = pd.DataFrame(columns=corr_dat.columns)
    elif (corr_dat['Pearson'] <= 0).all():
        top_positive_pearson = pd.DataFrame(columns=corr_dat.columns)
        top_negative_pearson = corr_dat.tail(10)
    else:
        top_positive_pearson = corr_dat[corr_dat['Pearson'] > 0].head(10)
        top_negative_pearson = corr_dat[corr_dat['Pearson'] < 0].tail(10)

    top_corr = pd.concat([top_positive_pearson, top_negative_pearson])
    if top_corr.empty:
        raise ValueError("No features to plot in top_corr.")
    
    plt.figure(figsize=(10, len(top_corr) * 0.5))
    sns.barplot(x=top_corr['Pearson'], y=top_corr.index)
    plt.title('Top 10 Positive and Negative Pearson Correlations with ust')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, pearson_plot_filename))
    plt.close()

    plt.figure(figsize=(10, len(top_corr) * 0.5))
    sns.barplot(x=top_corr['Spearman'], y=top_corr.index)
    plt.title('Top 10 Positive and Negative Spearman Correlations with ust')
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, spearman_plot_filename))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=corr_dat['Pearson'], y=corr_dat['Spearman'])
    plt.title('Pearson vs Spearman Correlations with ust')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Spearman Correlation')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    for i, feature in enumerate(corr_dat.index):
        if abs(corr_dat['Pearson'][i] - corr_dat['Spearman'][i]) > 0.1:
            plt.text(corr_dat['Pearson'][i], corr_dat['Spearman'][i], feature, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, scatter_plot_filename))
    plt.close()

if __name__ == "__main__":
    output_dir = "output_data"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        exit(1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dat = data_loading_fsk_v1()
    if raw_dat is not None:
        logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
        cleaned_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        if cleaned_dat is not None:
            logger.info(f"Cleaned DataFrame Shape: {cleaned_dat.shape}")
            transformed_dat = data_transforming_fsk_v1(cleaned_dat)
            if transformed_dat is not None:
                logger.info(f"Transformed DataFrame Shape: {transformed_dat.shape}")
                scale_clean_engineer_dat = data_engineering_fsk_v1(transformed_dat)
                if scale_clean_engineer_dat is not None:
                    logger.info(f"Features Engineered DataFrame Shape: {scale_clean_engineer_dat.shape}")
                    cleaned_engineered_dat = data_preprocessing(scale_clean_engineer_dat)
                    if cleaned_engineered_dat is not None:
                        logger.info(f"Cleaned Engineered DataFrame Shape (After Outlier Removal): {cleaned_engineered_dat.shape}")
                        cleaned_engineered_dat.to_csv(
                                os.path.join(output_dir, f"cleaned_scale_clean_engineer_dat_{timestamp}.csv"),
                                index=False,
                                encoding='utf-8-sig'
                            )
                        logger.info(f"Saved cleaned engineered data to {output_dir}/cleaned_scale_clean_engineer_dat_{timestamp}.csv")
                        corr_dat = compute_correlations(cleaned_engineered_dat)
                        if corr_dat is not None:
                            corr_dat.to_csv(
                                                os.path.join('output_data', f"correlations_with_ust_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                                                index=True,
                                                encoding='utf-8-sig'
                                                )
                            logger.info("Saved correlations to 'correlations_with_ust.csv'.")
                            logger.info("Correlations with ust (Pearson and Spearman):")
                            logger.info(f"\n{corr_dat}")
                            
                            visualize_correlations(corr_dat)
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