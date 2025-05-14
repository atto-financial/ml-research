import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from app.data.data_engineering import engineer_features
from app.data.data_cleaning import data_cleaning, transform_data, load_data_fsk_v1
import os

def compute_correlations(engineered_data: pd.DataFrame) -> pd.DataFrame:
    
    try:
        if engineered_data is None or engineered_data.empty:
            print("Input DataFrame is None or empty.")
            return None

        # Identify numeric columns (exclude user_id and ust)
        exclude_cols = ['user_id', 'ust']
        numeric_cols = [col for col in engineered_data.columns 
                       if col not in exclude_cols 
                       and engineered_data[col].dtype in [np.float64, np.int64]]

        # Compute Pearson and Spearman correlations with ust
        correlations = {}
        for col in numeric_cols:
            # Check for zero variance to avoid NaN correlations
            if engineered_data[col].std() == 0:
                print(f"Skipping {col}: Variance is zero (all values are the same).")
                correlations[col] = {'Pearson': np.nan, 'Spearman': np.nan}
                continue
            pearson_corr = engineered_data[col].corr(engineered_data['ust'], method='pearson')
            spearman_corr = engineered_data[col].corr(engineered_data['ust'], method='spearman')
            correlations[col] = {'Pearson': pearson_corr, 'Spearman': spearman_corr}

        # Create DataFrame for correlations
        corr_df = pd.DataFrame.from_dict(correlations, orient='index')
        corr_df = corr_df.sort_values(by='Pearson', ascending=False)

        # Save correlations to CSV
        corr_df.to_csv('correlations_with_ust.csv')
        print("Saved correlations to 'correlations_with_ust.csv'.")

        return corr_df

    except (ValueError, TypeError, KeyError) as e:
        print(f"Error during correlation computation: {str(e)}")
        return None

def visualize_correlations(corr_df: pd.DataFrame, output_dir: str = 'plots'):
    
    if corr_df is None or corr_df.empty:
        print("Correlation DataFrame is None or empty.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Remove NaN entries for visualization
    corr_df = corr_df.dropna()

    # Get top 10 positive and top 10 negative correlations based on Pearson
    top_positive_pearson = corr_df.head(10)
    top_negative_pearson = corr_df.tail(10)
    top_corr = pd.concat([top_positive_pearson, top_negative_pearson])

    # Bar plot for Pearson Correlation
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_corr['Pearson'], y=top_corr.index)
    plt.title('Top 10 Positive and Negative Pearson Correlations with ust (Default)')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pearson_correlations_with_ust.png')
    plt.close()

    # Bar plot for Spearman Correlation
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_corr['Spearman'], y=top_corr.index)
    plt.title('Top 10 Positive and Negative Spearman Correlations with ust (Default)')
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spearman_correlations_with_ust.png')
    plt.close()

    # Scatter plot to compare Pearson vs Spearman
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=corr_df['Pearson'], y=corr_df['Spearman'])
    plt.title('Pearson vs Spearman Correlations with ust')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Spearman Correlation')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    for i, feature in enumerate(corr_df.index):
        if abs(corr_df['Pearson'][i] - corr_df['Spearman'][i]) > 0.1:  # Highlight features with significant difference
            plt.text(corr_df['Pearson'][i], corr_df['Spearman'][i], feature, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pearson_vs_spearman.png')
    plt.close()

if __name__ == "__main__":
    # Load, clean, transform, engineer, remove outliers, and compute correlations
    raw_data = load_data_fsk_v1()
    if raw_data is not None:
        print("Raw DataFrame Shape:", raw_data.shape)
        cleaned_data = data_cleaning(raw_data, fill_lps='median', outlier_method='median')
        
        if cleaned_data is not None:
            print("\nCleaned DataFrame Shape:", cleaned_data.shape)
            transformed_data = transform_data(cleaned_data)
            
            if transformed_data is not None:
                print("\nTransformed DataFrame Shape:", transformed_data.shape)
                engineered_data = engineer_features(transformed_data)
                
                if engineered_data is not None:
                    print("\nEngineered DataFrame Shape:", engineered_data.shape)
                    
                    # Remove outliers
                    #engineered_data = remove_outliers(engineered_data)
                    
                    if engineered_data is not None:
                        print("\nCleaned Engineered DataFrame Shape (After Outlier Removal):", engineered_data.shape)
                        
                        # Compute correlations
                        corr_df = compute_correlations(engineered_data)
                        
                        if corr_df is not None:
                            print("\nCorrelations with ust (Pearson and Spearman):")
                            print(corr_df)
                            
                            # Visualize correlations
                            visualize_correlations(corr_df)
                            print("Correlation visualizations saved in 'plots' directory.")
                        else:
                            print("Failed to compute correlations.")
                    #else:
                        #print("Failed to remove outliers.")
                else:
                    print("Failed to engineer features.")
            else:
                print("Failed to transform data.")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")