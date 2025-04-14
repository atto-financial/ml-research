import pandas as pd
from sqlalchemy import create_engine, text
from app.config.settings import DB_URI

def load_data():
    engine = create_engine(DB_URI)
    query = text("""
        SELECT cdd_1 AS cdd1, cdd_2 AS cdd2, cdd_3 AS cdd3, cdd_4 AS cdd4, 
               cdd_5 AS cdd5, cdd_6 AS cdd6, cdd_7 AS cdd7, cdd_8 AS cdd8, 
               cdd_9 AS cdd9, cdd_10 AS cdd10, cdd_11 AS cdd11, 
               init_credit_score AS ins, user_status AS ust 
        FROM users 
        WHERE user_status = 1 OR (user_status = 0 and credit_score > 4)
    """)

    with engine.connect() as connection:
        raw_dat = pd.read_sql(query, connection)

    return raw_dat