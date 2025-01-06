import sqlite3
import pandas as pd
import json

class DataCleaning:
    def __init__(self, input_db_path=None, input_table_name=None, input_dataframe=None):
        """
        Initializes the DataCleaning class.

        :param input_db_path: Path to the SQLite database.
        :param input_table_name: Name of the table in the database.
        :param input_dataframe: Optional DataFrame to be used instead of a database.
        """
        self.input_db_path = input_db_path
        self.input_table_name = input_table_name
        self.dataframe = input_dataframe

    def load_data(self):
        """
        Loads data from the SQLite database into a DataFrame if a database path is provided.
        If a DataFrame is already provided, it will be used directly.
        """
        if self.input_db_path and self.input_table_name:
            # Establish a connection to the database
            conn = sqlite3.connect(self.input_db_path)

            # Read the data into a DataFrame
            self.dataframe = pd.read_sql_query(f"SELECT * FROM {self.input_table_name}", conn)

            # Close the connection
            conn.close()
        elif self.dataframe is not None:
            print("Using the provided DataFrame.")
        else:
            raise ValueError("No database path or DataFrame provided. Cannot load data.")

    def convert_to_datetime(self, column_name, dayfirst=True):
        """Converts a column to datetime format."""
        if self.dataframe is not None:
            self.dataframe[column_name] = pd.to_datetime(self.dataframe[column_name], dayfirst=dayfirst)
        else:
            raise ValueError("Dataframe is not loaded. Use load_data() first.")

    def convert_to_numeric(self, columns, dtype=float):
        """Converts specified columns to numeric type, coercing errors to NaN."""
        if self.dataframe is not None:
            for col in columns:
                self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce').astype(dtype)
        else:
            raise ValueError("Dataframe is not loaded. Use load_data() first.")

    def apply_absolute_values(self, columns):
        """Applies absolute values to specified columns."""
        if self.dataframe is not None:
            self.dataframe[columns] = self.dataframe[columns].abs()
        else:
            raise ValueError("Dataframe is not loaded. Use load_data() first.")

    def replace_values(self, column_replacements):
        """Replaces values in specified columns based on replacement dictionaries."""
        if self.dataframe is not None:
            for column, replacements in column_replacements.items():
                self.dataframe[column] = self.dataframe[column].replace(replacements)
        else:
            raise ValueError("Dataframe is not loaded. Use load_data() first.")

    def convert_to_ordinal(self, column_name, order):
        """Converts a column to an ordinal variable based on a defined order."""
        if self.dataframe is not None:
            self.dataframe[column_name] = pd.Categorical(self.dataframe[column_name], categories=order, ordered=True)
        else:
            raise ValueError("Dataframe is not loaded. Use load_data() first.")

    def save_category_metadata(self, metadata, metadata_path):
        """Saves category metadata to a JSON file."""
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def get_dataframe(self):
        """Returns the cleaned DataFrame."""
        if self.dataframe is not None:
            return self.dataframe
        else:
            raise ValueError("Dataframe is not loaded. Use load_data() first.")

    def save_cleaned_data(self, cleaned_df, output_db_path, output_table_name):
        """
        Save the cleaned DataFrame back into a SQLite database.

        :param merged_df: The cleaned DataFrame to save.
        :param output_db_path: Path to the output SQLite database.
        :param output_table_name: Name of the table to save the cleaned data.
        """
        with sqlite3.connect(output_db_path) as conn:
            cleaned_df.to_sql(output_table_name, conn, if_exists="replace", index=False)

# Example usage
if __name__ == "__main__":
    # Path to the merged database
    merged_db_path = "data/merged_data.db"

    # Initialize the DataCleaning class
    cleaner = DataCleaning(input_db_path=merged_db_path, input_table_name="merged_data")

    # Load data
    cleaner.load_data()

    # Convert date column to datetime
    cleaner.convert_to_datetime("date", dayfirst=True)

    # Convert specified columns to numeric
    columns_to_convert = ['pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central',
                          'psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central']
    cleaner.convert_to_numeric(columns_to_convert)

    columns_to_convert2 = ['Daily Rainfall Total (mm)', 'Highest 30 Min Rainfall (mm)',
                           'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)',
                           'Min Temperature (deg C)', 'Maximum Temperature (deg C)',
                           'Min Wind Speed (km/h)', 'Max Wind Speed (km/h)']
    cleaner.convert_to_numeric(columns_to_convert2, dtype=float)

    # Apply absolute values to specified columns
    columns_to_abs = ['Max Wind Speed (km/h)', 'Wet Bulb Temperature (deg F)']
    cleaner.apply_absolute_values(columns_to_abs)

    # Replace identical words and abbreviations in lowercase words
    column_replacements = {
        'Dew Point Category': {
            'VH': 'very high', 'VERY HIGH': 'very high', 'Very High': 'very high',
            'L': 'low', 'Low': 'low', 'LOW': 'low',
            'High': 'high', 'HIGH': 'high', 'H': 'high', 'High Level': 'high',
            'Moderate': 'moderate', 'M': 'moderate', 'MODERATE': 'moderate',
            'Very Low': 'very low', 'VL': 'very low','Minimal':'very low', 'VERY LOW': 'very low',
            'Below Average': 'below average',
            'Extreme': 'extreme',
            'Normal': 'moderate'
        },
        'Wind Direction': {
            'W': 'west', 'W.': 'west', 'WEST': 'west',
            'E': 'east', 'E.': 'east', 'EAST': 'east',
            'S': 'south', 'Southward': 'south', 'SOUTH': 'south', 'S.': 'south',
            'NORTHEAST': 'northeast', 'NE':'northeast', 'NE.': 'northeast',
            'N': 'north', 'NORTH': 'north', 'Northward': 'north', 'N.': 'north',
            'NW': 'northwest', 'NORTHWEST': 'northwest', 'NW.': 'northwest',
            'SOUTHEAST': 'southeast', 'SE': 'southeast', 'SE.': 'southeast',
            'SW': 'southwest', 'SW.': 'southwest'
        }
    }
    cleaner.replace_values(column_replacements)

    # Define the order of the Dew Point Category
    dew_point_order = ['very low', 'low', 'below average', 'moderate', 'high', 'very high', 'extreme']

    # Convert Dew Point Category column to an ordinal variable
    cleaner.convert_to_ordinal('Dew Point Category', dew_point_order)

    effiency_order = ['Low', 'Medium', 'High']

    # Convert Daily Solar Panel Efficiency column to an ordinal variable
    cleaner.convert_to_ordinal('Daily Solar Panel Efficiency', effiency_order)

    # Save category metadata
    category_metadata = {
        'Dew Point Category': dew_point_order,
        'Daily Solar Panel Efficiency': effiency_order
    }
    cleaner.save_category_metadata(category_metadata, "data/category_metadata.json")

    # Get and display the cleaned DataFrame
    cleaned_df = cleaner.get_dataframe()

    output_db_path = "data/cleaned_data.db"  # Path to save the cleaned data
    output_table_name = "cleaned_data"  # Table name for cleaned data
    # Save the cleaned data back into a database
    cleaner.save_cleaned_data(cleaned_df, output_db_path, output_table_name)

    print(cleaned_df.head())