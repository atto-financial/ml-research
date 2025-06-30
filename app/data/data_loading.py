import pandas as pd
import numpy as np
import logging
from typing import Optional
from app.utils.db_connection import get_db_connection


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def data_loading_set_v1()-> Optional[pd.DataFrame]:
    query = """
        SELECT 
            c.set_1 AS set1, c.set_2 AS set2, c.set_3 AS set3, c.set_4 AS set4, 
            c.set_5 AS set5, c.set_6 AS set6, c.set_7 AS set7, c.set_8 AS set8, 
            c.set_9 AS set9, c.set_10 AS set10, c.set_11 AS set11, 
            u.latest_loan_payoff_score AS ins, u.user_status AS ust 
        FROM 
            set_answers AS c
        INNER JOIN 
            users AS u ON c.user_id = u.id
        WHERE 
            u.user_status = 1 
            OR (u.user_status = 0 AND u.payoff_score > 4);
        """
    try:
        conn = get_db_connection() 
        raw_dat = pd.read_sql(query, conn)
        logger.info(f"Load data completed")
        return raw_dat
    except Exception as e:
        logger.error(f"Error while executing query: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()


def data_loading_fsk_v1() -> Optional[pd.DataFrame]:

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
        raw_dat = pd.read_sql(query, conn)
        logger.info(f"Load raw data completed")

        raw_dat['ust'] = raw_dat['ust'].replace([np.inf, -np.inf], np.nan)  
        raw_dat['ust'] = raw_dat['ust'].fillna(0).astype(np.int64)
        
        return raw_dat
    
    except Exception as e:
        logger.error(f"Error while executing query: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    logger.info("Starting data loading process")
    raw_dat = data_loading_fsk_v1()
    if raw_dat is not None:
        logger.info(f"Raw DataFrame Shape: {raw_dat.shape}")
        logger.info(f"Raw DataFrame Columns: {raw_dat.columns.tolist()}")
        logger.info(f"Raw DataFrame Info:\n{raw_dat.info()}")
        logger.info(f"Raw DataFrame Sample:\n{raw_dat.sample(5).to_string()}")
    else:
        logger.error("Failed to load data.")