import pandas as pd
import numpy as np
import logging
from typing import Optional, Union
from app.utils.db_connection import get_db_connection
from typing import Optional, Union, Iterator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def data_loading_fsk_v1(chunksize: Optional[int] = None) -> Optional[Union[pd.DataFrame, Iterator[pd.DataFrame]]]:
   
    query = """
        SELECT 
            f.fht_1 AS fht1, 
            f.fht_2 AS fht2, 
            f.fht_3 AS fht3, 
            f.fht_4 AS fht4, 
            f.fht_5 AS fht5, 
            f.fht_6 AS fht6, 
            f.fht_7 AS fht7, 
            f.fht_8 AS fht8, 
            f.set_9 AS set1, 
            f.set_10 AS set2, 
            f.kmsi_1 AS kmsi1, 
            f.kmsi_2 AS kmsi2, 
            f.kmsi_3 AS kmsi3, 
            f.kmsi_4 AS kmsi4, 
            f.kmsi_5 AS kmsi5, 
            f.kmsi_6 AS kmsi6, 
            f.kmsi_7 AS kmsi7, 
            f.kmsi_8 AS kmsi8,
            u.user_status AS ust,
            u.id AS user_id
        FROM 
            fsk_answers AS f
        INNER JOIN 
            users AS u ON f.user_id = u.id
        WHERE 
            (u.user_status = 1 AND u.user_verified = 3 AND f.feature_label = "fsk_v2.0") OR 
            (u.user_status = 0 AND u.user_verified = 3 AND u.payoff_score >= 5 AND f.feature_label = "fsk_v2.0");
    """
    try:
        conn = get_db_connection()
        numerical_columns = [
            'fht1', 'fht2', 'fht3', 'fht4', 'fht5', 'fht6', 'fht7', 'fht8',
            'set1', 'set2', 'kmsi1', 'kmsi2', 'kmsi3', 'kmsi4', 'kmsi5', 'kmsi6', 'kmsi7', 'kmsi8'
        ]
        categorical_columns = ['ust']
        
        if chunksize:
            logger.info(f"Loading data in chunks of size {chunksize}")
            return pd.read_sql(query, conn, chunksize=chunksize)
        else:
            logger.info("Loading full dataset")
            raw_dat = pd.read_sql(query, conn)
            
            for col in numerical_columns:
                raw_dat[col] = raw_dat[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.int64)
            
            for col in categorical_columns:
                raw_dat[col] = raw_dat[col].astype('category')
            
            return raw_dat
    
    except Exception as e:
        logger.error(f"Error while executing query: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    logger.info("Starting data loading process")
    raw_dat = data_loading_fsk_v1(chunksize=None)
    if raw_dat is not None:
        if isinstance(raw_dat, pd.DataFrame):
            logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
            logger.info(f"Raw DataFrame Columns: {raw_dat.columns.tolist()}")
            logger.info(f"Raw DataFrame Dtypes: {raw_dat.dtypes.to_dict()}")
            logger.info(f"Raw DataFrame Sample (5 rows):\n{raw_dat.sample(5).to_string()}")
        else:
            logger.info("Data loaded as chunk iterator")
    else:
        logger.error("Failed to load data.")