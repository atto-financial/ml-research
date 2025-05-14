import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from sklearn.preprocessing import StandardScaler
from .data_cleaning import data_cleaning, transform_data, load_data_fsk_v1
import os

def engineer_features(transformed_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    
    try:
        if transformed_data is None or transformed_data.empty:
            print("Input DataFrame is None or empty.")
            return None

        print("Columns in Transformed DataFrame:", transformed_data.columns.tolist())

        engineered_data = transformed_data.copy()

        # Define groups based on behavioral economics categories
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

        # Check if all required columns exist
        missing_cols = []
        for group, cols in groups.items():
            for col in cols:
                if col not in engineered_data.columns:
                    missing_cols.append(col)
        if missing_cols:
            print(f"Error: Missing columns in DataFrame: {missing_cols}")
            return None

        # 1. aggregate features (sum and mean) for each group
        for group, cols in groups.items():
            engineered_data[f'{group}_sum'] = engineered_data[cols].sum(axis=1)
            engineered_data[f'{group}_mean'] = engineered_data[cols].mean(axis=1)

        # 2. ratio features to capture relationships
        engineered_data['spending_to_saving_ratio'] = engineered_data['spending_sum'] / (engineered_data['saving_sum'] + 1e-6)
        engineered_data['debt_to_paying_off_ratio'] = engineered_data['debt_sum'] / (engineered_data['paying_off_sum'] + 1e-6)
        engineered_data['avoidance_to_vigilance_ratio'] = engineered_data['avoidance_sum'] / (engineered_data['vigilance_sum'] + 1e-6)
        engineered_data['worship_to_status_ratio'] = engineered_data['worship_sum'] / (engineered_data['status_sum'] + 1e-6)

        # 3. interaction terms to reflect complex behaviors
        engineered_data['spending_avoidance_interaction'] = engineered_data['spending_mean'] * engineered_data['avoidance_mean']
        engineered_data['saving_planning_interaction'] = engineered_data['saving_mean'] * engineered_data['planning_mean']
        engineered_data['debt_vigilance_interaction'] = engineered_data['debt_mean'] * engineered_data['vigilance_mean']
        engineered_data['paying_off_worship_interaction'] = engineered_data['paying_off_mean'] * engineered_data['worship_mean']

        # 4. Additional features from lps
        # engineered_data['lps_negative'] = (engineered_data['lps'] < 0).astype(int)
        # engineered_data['lps_absolute'] = engineered_data['lps'].abs()

        # 5. Handle infinite or NaN values from division
        engineered_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        engineered_data.fillna(0, inplace=True)

        return engineered_data

    except (ValueError, TypeError, KeyError) as e:
        print(f"Error during feature engineering: {str(e)}")
        return None

# def remove_outliers(engineered_data: pd.DataFrame) -> Optional[pd.DataFrame]:
   
#     try:
#         if engineered_data is None or engineered_data.empty:
#             print("Input Engineered DataFrame is None or empty.")
#             return None

#         cleaned_data = engineered_data.copy()
#         initial_rows = len(cleaned_data)

#         # Identify numeric columns to check for outliers (exclude binary and identifier columns)
#         exclude_cols = ['user_id', 'ust']
#         numeric_cols = [col for col in cleaned_data.columns if col not in exclude_cols and cleaned_data[col].dtype in [np.float64, np.int64]]

#         # Apply IQR method to each numeric column
#         for col in numeric_cols:
#             Q1 = cleaned_data[col].quantile(0.25)
#             Q3 = cleaned_data[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR

#             # Filter out outliers
#             cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
#             print(f"Removed outliers in {col}: {initial_rows - len(cleaned_data)} rows removed.")

#             # Update initial rows for the next column
#             initial_rows = len(cleaned_data)

#         if cleaned_data.empty:
#             print("DataFrame is empty after removing outliers.")
#             return None

#         # Scale numerical features after removing outliers
#         numerical_cols = [col for col in cleaned_data.columns if col not in exclude_cols]
#         scaler = StandardScaler()
#         cleaned_data[numerical_cols] = scaler.fit_transform(cleaned_data[numerical_cols])

#         return cleaned_data

#     except (ValueError, TypeError, KeyError) as e:
#         print(f"Error during outlier removal: {str(e)}")
#         return None

# def visualize_engineered_data(cleaned_data: pd.DataFrame, output_dir: str = 'plots'):
 
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Correlation matrix to identify feature relationships
#     corr = cleaned_data.drop(columns=['user_id']).corr()
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(corr, annot=False, cmap='coolwarm')
#     plt.title('Correlation Matrix of Cleaned Features')
#     plt.savefig(f'{output_dir}/correlation_matrix.png')
#     plt.close()

#     # Distribution plots for key features
#     key_features = ['spending_mean', 'saving_mean', 'debt_mean', 'avoidance_mean', 'vigilance_mean']
#     for feature in key_features:
#         if feature in cleaned_data.columns:
#             plt.figure(figsize=(8, 6))
#             sns.histplot(cleaned_data[feature], bins=20, kde=True)
#             plt.title(f'Distribution of {feature} (After Outlier Removal)')
#             plt.savefig(f'{output_dir}/{feature}_distribution.png')
#             plt.close()

#     # Boxplot to compare features by ust (target variable)
#     if 'ust' in cleaned_data.columns:
#         for feature in key_features:
#             if feature in cleaned_data.columns:
#                 plt.figure(figsize=(8, 6))
#                 sns.boxplot(x='ust', y=feature, data=cleaned_data)
#                 plt.title(f'{feature} by Risk Status (ust)')
#                 plt.savefig(f'{output_dir}/{feature}_by_ust.png')
#                 plt.close()

#     # Scatter plot of lps vs key features
#     for feature in key_features:
#         if feature in cleaned_data.columns:
#             plt.figure(figsize=(8, 6))
#             sns.scatterplot(x='lps', y=feature, hue='ust', data=cleaned_data)
#             plt.title(f'{feature} vs lps (Colored by ust)')
#             plt.savefig(f'{output_dir}/{feature}_vs_lps.png')
#             plt.close()

if __name__ == "__main__":
    # Load, clean, transform, engineer, remove outliers, and train model
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
                    print("\nFeatures Engineered DataFrame Shape:", engineered_data.shape)
                    print("\nColumns in Features Engineered DataFrame:", engineered_data.columns.tolist())
                    print("\nFeatures Engineered DataFrame:",engineered_data.head())
                    print("\nSummary Statistics:", engineered_data.describe())
                    
                    # Remove outliers
                    # cleaned_engineered_data = remove_outliers(engineered_data)
                    
                    # if cleaned_engineered_data is not None:
                    #     print("\nCleaned Engineered DataFrame Shape (After Outlier Removal):", cleaned_engineered_data.shape)
                    #     print(cleaned_engineered_data.head())
                    #     print("\nSummary Statistics:")
                    #     print(cleaned_engineered_data.describe())
                        
                        # Visualize the cleaned data
                        # visualize_engineered_data(cleaned_engineered_data)
                        # print("Visualizations saved in 'plots' directory.")
                        
                    # else:
                    #     print("Failed to remove outliers.")
                else:
                    print("Failed to engineer features.")
            else:
                print("Failed to transform data.")
        else:
            print("Failed to clean data.")
    else:
        print("Failed to load data.")