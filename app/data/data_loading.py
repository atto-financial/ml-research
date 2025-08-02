import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, Iterator
from app.utils.db_connection import get_db_connection, test_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def data_loading_fsk_v1(chunksize: Optional[int] = None) -> Optional[Union[pd.DataFrame, Iterator[pd.DataFrame]]]:
    # Test database connection
    if not test_db_connection():
        logger.error("Cannot connect to database")
        return None

    # SQL query
    query = """
    select
    u.user_id,
    a.answer,
    case
    	when l.loan_status = 'npl' then 1
    	when l.loan_status = 'healthy' then 0
    end ust
    FROM
        users u
    LEFT JOIN answers a ON a.user_id = u.user_id
    LEFT JOIN features f ON f.feature_id = a.feature_id
    LEFT JOIN loan_summary_statuses l ON l.user_id = u.user_id
    WHERE f.feature_slug IS NOT NULL 
        AND a.answer IS NOT NULL 
        AND ((l.loan_status in ('npl') and l.payoff_score < 0) OR (l.loan_status in ('healthy') and l.payoff_score >= 5))
        and feature_slug = 'fsk_v2.0'
    """

    try:
        conn = get_db_connection()
        # numerical_columns = [
        #     'fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8',
        #     'set1', 'set2', 'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8'
        # ]
        numerical_columns = [
            'fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8',
             'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8'
        ]
        categorical_columns = ['ust']

        def process_chunk(chunk):
            # Apply extract_db_data to the chunk
            processed_chunk = extract_db_data(chunk)

            # Verify columns exist
            missing_cols = [col for col in numerical_columns +
                            categorical_columns if col not in processed_chunk.columns]
            if missing_cols:
                logger.error(f"Missing columns in chunk: {missing_cols}")
                return None

            # Apply type conversions
            for col in numerical_columns:
                processed_chunk[col] = processed_chunk[col].replace(
                    [np.inf, -np.inf], np.nan).fillna(0).astype(np.int64)
            for col in categorical_columns:
                processed_chunk[col] = processed_chunk[col].astype('category')
            return processed_chunk

        if chunksize:
            logger.info(f"Loading data in chunks of size {chunksize}")
            raw_chunks = pd.read_sql(query, conn, chunksize=chunksize)
            processed_chunks = []

            for chunk in raw_chunks:
                result = process_chunk(chunk)
                if result is not None:
                    processed_chunks.append(result)

            # Combine chunks into a single DataFrame
            if not processed_chunks:
                logger.warning(
                    "No valid chunks after processing; returning empty DataFrame")
                return pd.DataFrame(columns=numerical_columns + categorical_columns + ['user_id', 'answer', 'ust'])
            return pd.concat(processed_chunks, ignore_index=True)
        else:
            logger.info("Loading full dataset")
            plain_dat = pd.read_sql(query, conn)
            raw_dat = extract_db_data(plain_dat)

            # Verify columns exist
            missing_cols = [col for col in numerical_columns +
                            categorical_columns if col not in raw_dat.columns]
            if missing_cols:
                logger.error(f"Missing columns in DataFrame: {missing_cols}")
                return pd.DataFrame(columns=numerical_columns + categorical_columns + ['user_id', 'answer', 'ust'])

            # Apply type conversions
            for col in numerical_columns:
                raw_dat[col] = raw_dat[col].replace(
                    [np.inf, -np.inf], np.nan).fillna(0).astype(np.int64)
            for col in categorical_columns:
                raw_dat[col] = raw_dat[col].astype('category')
            return raw_dat

    except Exception as e:
        logger.error(f"Error while executing query: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()


def extract_db_data(data: pd.DataFrame) -> pd.DataFrame:
    # attr_mapping = [
    #     ('fht1', 'fht', 1), ('fht2', 'fht', 2), ('fht3', 'fht', 3), ('fht4', 'fht', 4),
    #     ('fht5', 'fht', 5), ('fht6', 'fht', 6), ('fht7', 'fht', 7), ('fht8', 'fht', 8),
    #     ('set1', 'set', 9), ('set2', 'set', 10),
    #     ('kmsi1', 'kmsi', 11), ('kmsi2', 'kmsi', 12), ('kmsi3', 'kmsi', 13), ('kmsi4', 'kmsi', 14),
    #     ('kmsi5', 'kmsi', 15), ('kmsi6', 'kmsi', 16), ('kmsi7', 'kmsi', 17), ('kmsi8', 'kmsi', 18)
    # ]
    attr_mapping = [
        ('fht1', 'fht', 1), ('fht2', 'fht', 2), ('fht3', 'fht', 3), ('fht4', 'fht', 4),
        ('fht5', 'fht', 5), ('fht6', 'fht', 6), ('fht7', 'fht', 7), ('fht8', 'fht', 8),
        ('kmsi1', 'kmsi', 11), ('kmsi2', 'kmsi', 12), ('kmsi3', 'kmsi', 13), ('kmsi4', 'kmsi', 14),
        ('kmsi5', 'kmsi', 15), ('kmsi6', 'kmsi', 16), ('kmsi7', 'kmsi', 17), ('kmsi8', 'kmsi', 18)
    ]

    # Initialize columns
    for attr, _, _ in attr_mapping:
        data[attr] = None

    filtered_indices = []
    for index, dat in data.iterrows():
        try:
            # Parse 'answer' (list of dicts)
            answer_data = dat['answer']
            if not isinstance(answer_data, list):
                logger.error(f"Answer is not a list at index {index}: {answer_data}")
                continue
            if len(answer_data) < 18:
                logger.error(f"Answer array has fewer than 18 elements at index {index}: {len(answer_data)}")
                continue

            # Create a mapping of questionNumber to choiceNumber
            question_to_choice = {}
            for item in answer_data:
                if not isinstance(item, dict) or 'group' not in item or 'choiceNumber' not in item or 'questionNumber' not in item:
                    logger.error(f"Invalid answer item format at index {index}: {item}")
                    continue
                try:
                    question_num = int(item['questionNumber'])
                    choice = int(item['choiceNumber'])
                    question_to_choice[question_num] = (item['group'], choice)
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid questionNumber or choiceNumber at index {index}: {item}")
                    continue

            # Map answers to columns
            valid_row = True
            for attr, expected_group, question_num in attr_mapping:
                if question_num in question_to_choice:
                    group, choice = question_to_choice[question_num]
                    if group == expected_group:
                        data.at[index, attr] = choice
                    else:
                        logger.error(f"Mismatched group at index {index}, question {question_num}: expected {expected_group}, got {group}")
                        valid_row = False
                        break
                else:
                    logger.error(f"Missing question {question_num} at index {index}")
                    valid_row = False
                    break

            if valid_row:
                filtered_indices.append(index)

        except (KeyError, TypeError, IndexError) as e:
            logger.error(f"Error processing row at index {index}: {e}")
            continue

    # Define columns to keep (excluding 'answer')
    columns_to_keep = [attr for attr, _, _ in attr_mapping] + ['ust', 'user_id']

    # Return filtered DataFrame without 'answer'
    if not filtered_indices:
        logger.warning("No valid rows after processing; returning empty DataFrame")
        return pd.DataFrame(columns=columns_to_keep)
    
    return data.loc[filtered_indices, columns_to_keep].copy()


if __name__ == "__main__":
    logger.info("Starting data loading process")
    raw_dat = data_loading_fsk_v1(chunksize=None)
    if raw_dat is not None:
        if isinstance(raw_dat, pd.DataFrame):
            logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
            logger.info(f"Raw DataFrame Columns: {raw_dat.columns.tolist()}")
            logger.info(f"Raw DataFrame Dtypes: {raw_dat.dtypes.to_dict()}")
            logger.info(
                f"Raw DataFrame Sample (5 rows):\n{raw_dat.sample(5).to_string()}")
        else:
            logger.info("Data loaded as chunk iterator")
            for i, chunk in enumerate(raw_dat):
                if chunk is not None:
                    logger.info(f"Chunk {i+1} Shape: {chunk.shape}")
                    logger.info(
                        f"Chunk {i+1} Columns: {chunk.columns.tolist()}")
                    logger.info(
                        f"Chunk {i+1} Dtypes: {chunk.dtypes.to_dict()}")
                    logger.info(
                        f"Chunk {i+1} Sample (5 rows):\n{chunk.sample(5).to_string()}")
                else:
                    logger.error(f"Chunk {i+1} is None")
    else:
        logger.error("Failed to load data.")