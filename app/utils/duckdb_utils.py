import duckdb
import pandas as pd
import os
import logging
from app.utils.db_connection import get_db_connection

logger = logging.getLogger(__name__)

DUCKDB_PATH = "artifacts/local_cache.duckdb"

def sync_to_duckdb(table_name: str, query: str):
    """
    Sync data from PostgreSQL to a local DuckDB file.
    """
    os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)
    
    try:
        # Load from PG
        conn_pg = get_db_connection()
        logger.info(f"Fetching data from PostgreSQL for table: {table_name}")
        df = pd.read_sql(query, conn_pg)
        conn_pg.close()
        
        # Save to DuckDB
        conn_duck = duckdb.connect(DUCKDB_PATH)
        conn_duck.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        logger.info(f"Synced {len(df)} rows to DuckDB table: {table_name}")
        conn_duck.close()
        return True
    except Exception as e:
        logger.error(f"Failed to sync to DuckDB: {e}")
        return False

def query_duckdb(query: str) -> pd.DataFrame:
    """
    Query the local DuckDB cache.
    """
    if not os.path.exists(DUCKDB_PATH):
        logger.warning("DuckDB cache file not found.")
        return pd.DataFrame()
        
    try:
        conn = duckdb.connect(DUCKDB_PATH)
        df = conn.execute(query).df()
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Failed to query DuckDB: {e}")
        return pd.DataFrame()
