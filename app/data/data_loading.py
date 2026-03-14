import os
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

    # Load SQL query from external file (which now points to dbt model)
    sql_file_path = os.path.join(os.path.dirname(__file__), 'queries', 'load_loan_data.sql')
    try:
        with open(sql_file_path, 'r') as f:
            query = f.read()
    except Exception as e:
        logger.error(f"Failed to read SQL file: {e}")
        return None

    try:
        conn = get_db_connection()
        
        if chunksize:
            logger.info(f"Loading data from dbt model in chunks of size {chunksize}")
            return pd.read_sql(query, conn, chunksize=chunksize)
        else:
            logger.info("Loading full dataset from dbt model")
            raw_dat = pd.read_sql(query, conn)
            logger.info(f"Loaded DataFrame with columns: {raw_dat.columns.tolist()}")
            return raw_dat

    except Exception as e:
        logger.error(f"Error while executing query: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()


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