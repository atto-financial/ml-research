import pandas as pd
from app.utils.db_connection import get_db_connection

# def load_data_cdd():
#     query = """
#         SELECT 
#             c.cdd_1 AS cdd1, c.cdd_2 AS cdd2, c.cdd_3 AS cdd3, c.cdd_4 AS cdd4, 
#             c.cdd_5 AS cdd5, c.cdd_6 AS cdd6, c.cdd_7 AS cdd7, c.cdd_8 AS cdd8, 
#             c.cdd_9 AS cdd9, c.cdd_10 AS cdd10, c.cdd_11 AS cdd11, 
#             u.latest_loan_payoff_score AS ins, u.user_status AS ust 
#         FROM 
#             cdd_answers AS c
#         INNER JOIN 
#             users AS u ON c.user_id = u.id
#         WHERE 
#             u.user_status = 1 
#             OR (u.user_status = 0 AND u.payoff_score > 4);
#     """

#     conn = get_db_connection()
#     try:
#         raw_dat = pd.read_sql(query, conn)
#     finally:
#         conn.close()

#     return raw_dat

def load_data_fsk_v1():
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
            f.cdd_9 AS cdd9, 
            f.cdd_10 AS cdd10, 
            f.kmsi_1 AS kmsi1, 
            f.kmsi_2 AS kmsi2, 
            f.kmsi_3 AS kmsi3, 
            f.kmsi_4 AS kmsi4, 
            f.kmsi_5 AS kmsi5, 
            f.kmsi_6 AS kmsi6, 
            f.kmsi_7 AS kmsi7, 
            f.kmsi_8 AS kmsi8,
            u.latest_loan_payoff_score AS ins, u.user_status AS ust 
        FROM 
            fck_answers AS f
        INNER JOIN 
            users AS u ON f.user_id = u.id
        WHERE 
            u.user_status = 1 
            OR (u.user_status = 0 AND u.payoff_score > 4);
    """

    conn = get_db_connection()
    try:
        raw_dat = pd.read_sql(query, conn)
    finally:
        conn.close()

    return raw_dat


if __name__ == "__main__":
    print("Loading data...")
    raw_fsk_v1 = load_data_fsk_v1()
    if raw_fsk_v1 is not None:
        print("\nDataFrame from load_data:")
        print(raw_fsk_v1)
        print("\nShape:", raw_fsk_v1.shape)
    else:
        print("Failed to load data.")
