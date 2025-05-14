import pandas as pd
import numpy as np
from typing import Optional
from .data_loading import load_data_fsk_v1
import matplotlib.pyplot as plt
import seaborn as sns
import os

def data_cleaning(raw_dat: pd.DataFrame, fill_lps: str = 'median', outlier_method: str = 'median') -> Optional[pd.DataFrame]:
    
    try:
        if raw_dat is None or raw_dat.empty:
            print("Input DataFrame is None or empty.")
            return None

        clean_dat = raw_dat.copy()
        initial_rows = len(clean_dat)

        # Remove duplicates
        clean_dat = clean_dat.drop_duplicates()
        print(f"Removed {initial_rows - len(clean_dat)} duplicate rows.")

        # Define column types
        categorical_columns = [
            'fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8',
            'set9', 'set10', 'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8'
        ]
        numeric_columns = ['lps']
        
        # Convert to numeric and handle 'ust'
        for col in categorical_columns + numeric_columns:
            clean_dat[col] = pd.to_numeric(clean_dat[col], errors='coerce')
        clean_dat['ust'] = pd.to_numeric(clean_dat['ust'], errors='coerce').astype('Int64')

        # Handle NaNs in categorical columns (fill with mode)
        for col in categorical_columns:
            if clean_dat[col].isna().sum() > 0:
                mode_val = clean_dat[col].mode()[0]
                clean_dat[col] = clean_dat[col].fillna(mode_val)
                print(f"Filled {clean_dat[col].isna().sum()} NaNs in {col} with mode ({mode_val}).")

        # Handle NaNs in lps
        if fill_lps == 'median':
            fill_value = clean_dat['lps'].median()
        elif fill_lps == 'mean':
            fill_value = clean_dat['lps'].mean()
       
        clean_dat['lps'] = clean_dat['lps'].fillna(fill_value)
        print(f"Filled NaNs in 'lps' with {fill_lps} ({fill_value}).")

        # Drop rows with NaN in other columns
        initial_rows_with_na = len(clean_dat)
        clean_dat = clean_dat.dropna()
        print(f"Dropped {initial_rows_with_na - len(clean_dat)} rows with missing values.")

        # Validate categorical columns
        for col in categorical_columns:
            valid_values = [1, 2, 3]  # Adjust based on domain knowledge
            invalid = clean_dat[~clean_dat[col].isin(valid_values)][col]
            if not invalid.empty:
                mode_val = clean_dat[col].mode()[0]
                clean_dat.loc[~clean_dat[col].isin(valid_values), col] = mode_val
                print(f"Replaced {len(invalid)} invalid values in {col} with mode ({mode_val}).")

        # Handle outliers in numeric columns
        for col in numeric_columns:
            Q1 = clean_dat[col].quantile(0.25)
            Q3 = clean_dat[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = clean_dat[(clean_dat[col] < lower_bound) | (clean_dat[col] > upper_bound)][col]
            if not outliers.empty:
                if outlier_method == 'median':
                    clean_dat.loc[(clean_dat[col] < lower_bound) | (clean_dat[col] > upper_bound), col] = clean_dat[col].median()
                elif outlier_method == 'cap':
                    clean_dat[col] = clean_dat[col].clip(lower=lower_bound, upper=upper_bound)
                elif outlier_method == 'remove':
                    clean_dat = clean_dat[(clean_dat[col] >= lower_bound) & (clean_dat[col] <= upper_bound)]
                print(f"Handled {len(outliers)} outliers in {col} using {outlier_method} method.")

        if clean_dat.empty:
            print("DataFrame is empty after cleaning.")
            return None

        return clean_dat

    except (ValueError, TypeError, KeyError) as e:
        print(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    raw_data = load_data_fsk_v1()
    if raw_data is not None:
        print("Raw DataFrame Shape:", raw_data.shape)
        print(raw_data.head())

        cleaned_data = data_cleaning(raw_data, fill_lps='median', outlier_method='median')
        if cleaned_data is not None:
            print("\nCleaned DataFrame Shape:", cleaned_data.shape)
            print(cleaned_data.head())
            print("\nSummary Statistics:")
            print(cleaned_data.describe())
            
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")


def transform_data(cleaned_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    
    try:
        if cleaned_data is None or cleaned_data.empty:
            print("Input DataFrame is None or empty.")
            return None

        transformed_data = cleaned_data.copy()

        # Define transformation mappings
        fht_mapping = {1: 3, 2: 2, 3: 1}      # fht1-8: 1->3, 2->2, 3->1
        set_mapping = {1: 1, 2: 2, 3: 3}      # set9, set10: 1->1, 2->2, 3->3
        kmsi16_mapping = {1: 1, 2: 3}         # kmsi1-6: 1->1, 2->3
        kmsi78_mapping = {1: 3, 2: 1}         # kmsi7-8: 1->3, 2->1

        # Define columns
        fht_columns = ['fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8']
        set_columns = ['set9', 'set10']
        kmsi16_columns = ['kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6']
        kmsi78_columns = ['kmsi7', 'kmsi8']

        # Filter columns that exist in DataFrame
        fht_columns = [col for col in fht_columns if col in transformed_data.columns]
        set_columns = [col for col in set_columns if col in transformed_data.columns]
        kmsi16_columns = [col for col in kmsi16_columns if col in transformed_data.columns]
        kmsi78_columns = [col for col in kmsi78_columns if col in transformed_data.columns]

        # Transform fht columns
        for col in fht_columns:
            if transformed_data[col].isin(fht_mapping.keys()).all():
                transformed_data[col] = transformed_data[col].map(fht_mapping)
                print(f"Transformed values in {col} using fht_mapping.")
            else:
                print(f"Warning: {col} contains invalid values not in {list(fht_mapping.keys())}.")

        # Transform set columns
        for col in set_columns:
            if transformed_data[col].isin(set_mapping.keys()).all():
                transformed_data[col] = transformed_data[col].map(set_mapping)
                print(f"Validated values in {col} using set_mapping.")
            else:
                print(f"Warning: {col} contains invalid values not in {list(set_mapping.keys())}.")

        # Transform kmsi1-6 columns
        for col in kmsi16_columns:
            if transformed_data[col].isin(kmsi16_mapping.keys()).all():
                transformed_data[col] = transformed_data[col].map(kmsi16_mapping)
                print(f"Transformed values in {col} using kmsi16_mapping.")
            else:
                print(f"Warning: {col} contains invalid values not in {list(kmsi16_mapping.keys())}.")

        # Transform kmsi7-8 columns
        for col in kmsi78_columns:
            if transformed_data[col].isin(kmsi78_mapping.keys()).all():
                transformed_data[col] = transformed_data[col].map(kmsi78_mapping)
                print(f"Transformed values in {col} using kmsi78_mapping.")
            else:
                print(f"Warning: {col} contains invalid values not in {list(kmsi78_mapping.keys())}.")

        return transformed_data

    except (ValueError, TypeError, KeyError) as e:
        print(f"Error during transformation: {str(e)}")
        return None

def visualize_transformed_data(original_data: pd.DataFrame, transformed_data: pd.DataFrame, output_dir: str = 'plots'):

    os.makedirs(output_dir, exist_ok=True)
    
    columns_to_plot = [
        'fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8',
        'set9', 'set10',
        'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8'
    ]
    columns_to_plot = [col for col in columns_to_plot if col in transformed_data.columns]

    for col in columns_to_plot:
        plt.figure(figsize=(10, 5))
        
        # Before transformation
        plt.subplot(1, 2, 1)
        sns.countplot(x=original_data[col])
        plt.title(f'{col} Before Transformation')
        
        # After transformation
        plt.subplot(1, 2, 2)
        sns.countplot(x=transformed_data[col])
        plt.title(f'{col} After Transformation')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{col}_transformation_comparison.png')
        plt.close()

if __name__ == "__main__":
    # Load and clean data
    raw_data = load_data_fsk_v1()
    if raw_data is not None:
        print("Raw DataFrame Shape:", raw_data.shape)
        cleaned_data = data_cleaning(raw_data, fill_lps='median', outlier_method='median')
        
        if cleaned_data is not None:
            print("\nCleaned DataFrame Shape:", cleaned_data.shape)
            print(cleaned_data.head())
            
            # Transform data
            transformed_data = transform_data(cleaned_data)
            if transformed_data is not None:
                print("\nTransformed DataFrame Shape:", transformed_data.shape)
                print(transformed_data.head())
                print("\nSummary Statistics:")
                print(transformed_data.describe())
            else:
                print("Failed to transform data.")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")