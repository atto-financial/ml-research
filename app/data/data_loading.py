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
            c.cdd_1 AS cdd1, c.cdd_2 AS cdd2, c.cdd_3 AS cdd3, c.cdd_4 AS cdd4, 
            c.cdd_5 AS cdd5, c.cdd_6 AS cdd6, c.cdd_7 AS cdd7, c.cdd_8 AS cdd8, 
            c.cdd_9 AS cdd9, c.cdd_10 AS cdd10, c.cdd_11 AS cdd11, 
            u.latest_loan_payoff_score AS ins, u.user_status AS ust 
        FROM 
            cdd_answers AS c
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
            f.cdd_9 AS set1, 
            f.cdd_10 AS set2, 
            f.kmsi_1 AS kmsi1, 
            f.kmsi_2 AS kmsi2, 
            f.kmsi_3 AS kmsi3, 
            f.kmsi_4 AS kmsi4, 
            f.kmsi_5 AS kmsi5, 
            f.kmsi_6 AS kmsi6, 
            f.kmsi_7 AS kmsi7, 
            f.kmsi_8 AS kmsi8,
            u.user_blacklist AS ubl,
            u.user_status AS ust,
            u.id AS user_id
        FROM 
            fck_answers AS f
        INNER JOIN 
            users AS u ON f.user_id = u.id
        WHERE 
            (u.user_status = 1 AND u.user_verified = 3) 
            OR (u.user_status = 0 AND u.user_verified = 3 AND u.payoff_score >= 4)
            OR (u.user_blacklist =1 AND u.user_verified = 3);
    """
    try:
        conn = get_db_connection() 
        raw_dat = pd.read_sql(query, conn)
        logger.info(f"Load raw data completed")
        
        if 'ubl' in raw_dat.columns and 'ust' in raw_dat.columns:
            raw_dat.loc[(raw_dat['ubl'] == 1) & (raw_dat['ust'].isna()), 'ust'] = np.int64(1)
            raw_dat.drop(columns=['ubl'], inplace=True)
        else:
            print("Error: Columns 'ubl' or 'ust' not found in DataFrame")
        
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