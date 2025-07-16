import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Dict, Union, Iterator
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from .data_loading import data_loading_fsk_v1
from .data_cleaning import data_cleaning_fsk_v1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def map_columns(df: pd.DataFrame, columns: list, mapping: dict, col_name: str) -> pd.DataFrame:
   
    if not columns:
        logger.info(f"No {col_name} columns to map.")
        return df

    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.info(f"No valid {col_name} columns found in DataFrame.")
        return df

    for col in valid_columns:
        invalid_rows = ~df[col].isin(mapping.keys())
        if invalid_rows.any():
            df = df[~invalid_rows]
            logger.warning(f"Dropped {invalid_rows.sum()} rows with invalid values in {col}")
        df[col] = df[col].map(mapping).astype(np.float64)
    
    logger.info(f"Transformed values in {valid_columns} using {col_name} mapping to float64.")
    return df

def encode_categorical(df: pd.DataFrame, columns: list, encoding: str = 'onehot') -> pd.DataFrame:
    
    if not columns:
        logger.info("No categorical columns to encode.")
        return df

    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.info("No valid categorical columns found in DataFrame.")
        return df

    transformed_df = df.copy()
    for col in valid_columns:
        if transformed_df[col].isna().any():
            logger.warning(f"Found NaN values in {col}. Filling with 'missing'.")
            transformed_df[col] = transformed_df[col].fillna('missing')

        unique_values = transformed_df[col].unique()
        logger.info(f"Unique values in {col}: {unique_values}")

        if encoding == 'onehot':
            dummies = pd.get_dummies(transformed_df[col], prefix=col, dtype=np.float64)
            transformed_df = pd.concat([transformed_df.drop(col, axis=1), dummies], axis=1)
            logger.info(f"One-hot encoded {col} into {dummies.columns.tolist()}")
        elif encoding == 'label':
            le = LabelEncoder()
            transformed_df[col] = le.fit_transform(transformed_df[col]).astype(np.int64)
            logger.info(f"Label encoded {col} with {len(le.classes_)} classes")

    return transformed_df

def data_transforming_fsk_v1(
    clean_dat: Union[pd.DataFrame, Iterator[pd.DataFrame]], 
    metadata: Optional[Dict] = None
) -> Optional[pd.DataFrame]:
  
    try:
        # Initialize result list for chunked data
        result_dfs = []
        is_chunked = isinstance(clean_dat, Iterator)

        # Handle single DataFrame or chunks
        if not is_chunked:
            if clean_dat is None or clean_dat.empty:
                logger.error("Input DataFrame is None or empty.")
                return None
            clean_dat = [clean_dat]  # Treat as single chunk
        else:
            logger.info("Processing chunked input data.")

        for chunk_idx, chunk in enumerate(clean_dat):
            if chunk is None or chunk.empty:
                logger.warning(f"Chunk {chunk_idx} is None or empty. Skipping.")
                continue

            transform_dat = chunk.copy()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Check for zero-variance features
            exclude_cols = ['ust']
            numeric_cols = [col for col in transform_dat.columns 
                           if col not in exclude_cols and transform_dat[col].dtype in [np.float64, np.int64]]
            zero_variance_cols = [col for col in numeric_cols if transform_dat[col].var() == 0]
            if zero_variance_cols:
                logger.info(f"Found {len(zero_variance_cols)} features with zero variance: {zero_variance_cols}")

            # Drop unnecessary columns
            columns_to_drop = ['user_id']
            existing_columns = [col for col in columns_to_drop if col in transform_dat.columns]
            if existing_columns:
                transform_dat = transform_dat.drop(columns=existing_columns)
                logger.info(f"Dropped columns: {existing_columns}")

            # Define column mappings
            mappings = {
                'fht': {
                    'columns': ['fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8'],
                    'mapping': {1: 3, 2: 2, 3: 1}
                },
                'set': {
                    'columns': ['set1', 'set2'],
                    'mapping': {1: 1, 2: 2, 3: 3}
                },
                'kmsi16': {
                    'columns': ['kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6'],
                    'mapping': {1: 1, 2: 2, 3: 3}
                },
                'kmsi78': {
                    'columns': ['kmsi7', 'kmsi8'],
                    'mapping': {1: 3, 2: 2, 3: 1}
                }
            }

            # Apply mappings
            for key, config in mappings.items():
                transform_dat = map_columns(transform_dat, config['columns'], config['mapping'], key)

            # Handle NaN values in mapped columns
            mapped_columns = sum([config['columns'] for config in mappings.values()], [])
            mapped_columns = [col for col in mapped_columns if col in transform_dat.columns]
            if mapped_columns:
                nan_count = transform_dat[mapped_columns].isna().sum().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in mapped columns. Filling with medians.")
                    transform_dat[mapped_columns] = transform_dat[mapped_columns].fillna(
                        transform_dat[mapped_columns].median()
                    )

            # Handle categorical columns
            categorical_columns = []  
            categorical_encoding = 'onehot'  # Can be 'onehot' or 'label'
            if metadata and 'categorical_columns' in metadata:
                categorical_columns = metadata.get('categorical_columns', categorical_columns)
                categorical_encoding = metadata.get('categorical_encoding', categorical_encoding)

            transform_dat = encode_categorical(transform_dat, categorical_columns, categorical_encoding)

            # Handle 'ust' column
            if 'ust' in transform_dat.columns:
                invalid_ust = ~transform_dat['ust'].isin([0, 1])
                if invalid_ust.any():
                    logger.warning(f"Dropped {invalid_ust.sum()} rows with invalid values in ust")
                    transform_dat = transform_dat[~invalid_ust]
                transform_dat['ust'] = transform_dat['ust'].astype(np.int64)
                logger.info("Encoded ust column to int64.")

            result_dfs.append(transform_dat)
            logger.info(f"Processed chunk {chunk_idx} successfully.")

        if not result_dfs:
            logger.error("No valid chunks processed.")
            return None

        # Combine chunks
        final_df = pd.concat(result_dfs, ignore_index=True)
        logger.info("Data transformation completed successfully.")
        return final_df

    except Exception as e:
        logger.error(f"Error during transformation: {str(e)}")
        return None

if __name__ == "__main__":
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    
    raw_dat = data_loading_fsk_v1()
    if raw_dat is not None:
        clean_dat = data_cleaning_fsk_v1(raw_dat, outlier_method='median')
        if clean_dat is not None:
            # Example metadata
            metadata = {
                'categorical_columns': [],
                'categorical_encoding': 'onehot'
            }
            transform_dat = data_transforming_fsk_v1(clean_dat, metadata=metadata)
            if transform_dat is not None:
                logger.info(f"Transformed DataFrame Shape: {transform_dat.shape}")
                logger.info(f"Transformed DataFrame Columns: {transform_dat.columns.tolist()}")
                logger.info(f"Transformed DataFrame Sample:\n{transform_dat.sample(5).to_string()}")
            else:
                logger.error("Failed to transform data.")
        else:
            logger.error("Failed to clean data.")
    else:
        logger.error("Failed to load data.")