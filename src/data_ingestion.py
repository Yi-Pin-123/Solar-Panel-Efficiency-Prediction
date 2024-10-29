import sqlite3
import pandas as pd

class DataIngestion():
    def __init__ (self, db_path):
        self.db_path = db_path

    def get_df(self):
        # Establish a connection with the database using sqlite3
        conn = sqlite3.connect(self.db_path)

        # Obtain table name
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_name = cursor.fetchone()[0]        

        # Query to fetch the data and load the data into a pandas DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)

        cursor.close()
        conn.close()

        return df
  

