from backend.db import get_db

def get_table_names():
    """Get list of table names from the database"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            return [table[0] for table in tables]
    except Exception as e:
        return []
