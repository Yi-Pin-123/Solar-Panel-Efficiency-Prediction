import pandas as pd
import sqlite3

class DataMerger:
    def __init__(self, db1_path, db2_path, table1_name, table2_name, merge_column):
        """
        Initialize the DataMerger class.
        
        :param db1_path: Path to the first SQLite database.
        :param db2_path: Path to the second SQLite database.
        :param table1_name: Table name in the first database.
        :param table2_name: Table name in the second database.
        :param merge_column: Column name to merge the datasets on.
        """
        self.db1_path = db1_path
        self.db2_path = db2_path
        self.table1_name = table1_name
        self.table2_name = table2_name
        self.merge_column = merge_column

    def load_data(self):
        """
        Load data from the specified tables in the two databases.

        :return: DataFrames for the two tables.
        """
        with sqlite3.connect(self.db1_path) as conn1:
            df1 = pd.read_sql_query(f"SELECT * FROM {self.table1_name}", conn1)
        
        with sqlite3.connect(self.db2_path) as conn2:
            df2 = pd.read_sql_query(f"SELECT * FROM {self.table2_name}", conn2)
        
        return df1, df2

    def merge_remove_duplicate(self):
        """
        Merge the datasets on the specified column and remove duplicate rows.
        """
        df1, df2 = self.load_data()
        
        # Merge the datasets
        merged_df = pd.merge(df1, df2, on=self.merge_column, how="inner")
        
        # Drop duplicate rows
        merged_df.drop_duplicates(inplace=True)
        
        return merged_df

    def save_cleaned_data(self, merged_df, output_db_path, output_table_name):
        """
        Save the cleaned DataFrame back into a SQLite database.

        :param merged_df: The cleaned DataFrame to save.
        :param output_db_path: Path to the output SQLite database.
        :param output_table_name: Name of the table to save the cleaned data.
        """
        with sqlite3.connect(output_db_path) as conn:
            merged_df.to_sql(output_table_name, conn, if_exists="replace", index=False)

# Example usage
if __name__ == "__main__":
    # Replace with actual file paths and table names
    db1_path = "data/weather.db"
    db2_path = "data/air_quality.db"
    table1_name = "weather"
    table2_name = "air_quality"
    merge_column = "data_ref"
    output_db_path = "data/merged_data.db"  # Path to save the cleaned data
    output_table_name = "merged_data"  # Table name for cleaned data

    merger = DataMerger(db1_path, db2_path, table1_name, table2_name, merge_column)
    merged_data = merger.merge_remove_duplicate()

    # Save the cleaned data back into a database
    merger.save_cleaned_data(merged_data, output_db_path, output_table_name)

    # Display the cleaned data
    print(merged_data.head())
