import pandas as pd
from app.utils.db_connection import get_db_connection

def load_data():
    query = """
        SELECT cdd_1 AS cdd1, cdd_2 AS cdd2, cdd_3 AS cdd3, cdd_4 AS cdd4, 
               cdd_5 AS cdd5, cdd_6 AS cdd6, cdd_7 AS cdd7, cdd_8 AS cdd8, 
               cdd_9 AS cdd9, cdd_10 AS cdd10, cdd_11 AS cdd11, 
               init_credit_score AS ins, user_status AS ust 
        FROM users 
        WHERE user_status = 1 OR (user_status = 0 AND credit_score > 4)
    """

    conn = get_db_connection()
    try:
        raw_dat = pd.read_sql(query, conn)
    finally:
        conn.close()

    return raw_dat
