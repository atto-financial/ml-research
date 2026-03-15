import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from typing import List, Dict, Tuple

def raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Entry point for the dataframe."""
    return df

def numeric_cols(raw_data: pd.DataFrame) -> List[str]:
    """Identify numeric columns, excluding target."""
    exclude_cols = ['ust', 'user_id']
    return [col for col in raw_data.columns 
            if col not in exclude_cols 
            and raw_data[col].dtype in [np.float64, np.int64]]

def filtered_cols(raw_data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Drop columns based on specific suffixes to reduce noise."""
    cols_to_drop = [col for col in numeric_cols if col.endswith(('_sum', '_count', '_var'))]
    return raw_data.drop(columns=cols_to_drop)

def cleaned_data(filtered_cols: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Handle outliers for each numeric column using IQR median replacement."""
    df_out = filtered_cols.copy()
    # Filter numeric_cols to only those present in filtered_cols
    current_numeric = [c for c in numeric_cols if c in df_out.columns]
    
    for col in current_numeric:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
        if outliers_mask.any():
            df_out.loc[outliers_mask, col] = df_out[col].median()
            
    return df_out

def scaled_data(cleaned_data: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric columns and return the scaler object."""
    df = cleaned_data.copy()
    current_numeric = [c for c in numeric_cols if c in df.columns]
    
    scaler = StandardScaler()
    df[current_numeric] = scaler.fit_transform(df[current_numeric])
    
    # Drop zero variance columns after scaling
    zero_var = [col for col in current_numeric if df[col].var() == 0]
    if zero_var:
        df = df.drop(columns=zero_var)
        
    return df, scaler

def final_preprocessed_data(scaled_data: Tuple[pd.DataFrame, StandardScaler]) -> pd.DataFrame:
    """Return the final dataframe part of the tuple, rounded."""
    df = scaled_data[0].copy()
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].round(3)
    return df

def feature_scaler(scaled_data: Tuple[pd.DataFrame, StandardScaler]) -> StandardScaler:
    """Return the scaler part of the tuple."""
    return scaled_data[1]
