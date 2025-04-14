import pandas as pd
from app.utils.db_connection import get_db_connection

# def save_predictions_to_db(predictions_df):
#     conn = get_db_connection()
#     predictions_df.to_sql('predictions', conn, if_exists='append', index=False)
#     print("Predictions saved to database")