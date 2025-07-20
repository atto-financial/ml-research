import psycopg2
import logging
from app.config.settings import DB_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn


def test_db_connection():
    """
    Test the database connection by executing a simple query.
    Returns True if the connection is successful, False otherwise.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        logger.info("Database connection test successful: %s", result)
        cursor.close()
        conn.close()
        return True
    except psycopg2.Error as e:
        logger.error("Database connection test failed: %s", e)
        return False


if __name__ == "__main__":
    logger.info("Testing database connection")
    if test_db_connection():
        logger.info("Connection test passed")
    else:
        logger.error("Connection test failed")
