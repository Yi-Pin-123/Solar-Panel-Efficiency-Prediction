import pandas as pd
import numpy as np
from constants import *
class Cleaning:
    def __init__(self, df, numeric_cols, string_cols):
        self.df = df
        self.numeric_cols = numeric_cols
        self.string_cols = string_cols

        self.clean_data()  # Perform data cleaning
        self.check_data()  # Check data after cleaning

    def clean_data(self):
        self.df = self.clean_numerical_data()
        self.df = self.clean_string_data()
        self.df = self.clean_date_data()
        self.remove_outliers()
        self.clean_missing_data()  
        print("Data cleaned!")
        
    def clean_missing_data(self):
        self.df = self.impute_numerical_data()
        self.df = self.impute_string_data()
        return self.df

    def clean_numerical_data(self):
        try:
            for numeric_col in self.numeric_cols:
                # Convert the column to numeric, coercing errors to NaN
                self.df[numeric_col] = pd.to_numeric(self.df[numeric_col], errors='coerce')

                # Convert to integer if there are no NaN values
                if self.df[numeric_col].isnull().sum() == 0:
                    self.df[numeric_col] = self.df[numeric_col].astype(int)
        except Exception as e:
            print(f"An error occurred in cleaning numerical data: {e}")
        
        return self.df 

    def clean_string_data(self):
        # Replace placeholders with NaN
        self.df = self.df.replace(['-', '--'], np.nan)

        # Clean dew point categories
        def clean_dew_point(category):
            category = str(category).lower().strip()
            dew_point_map = {
                'vh': 'Very High', 'very high': 'Very High',
                'high': 'High', 'h': 'High',
                'moderate': 'Moderate', 'medium': 'Moderate', 'm': 'Moderate',
                'low': 'Low', 'l': 'Low',
                'very low': 'Very Low', 'vl': 'Very Low', 'below average': 'Very Low',
                'extreme': 'Extreme', 'minimal': 'Very Low', 'normal': 'Moderate'
            }
            return dew_point_map.get(category, np.nan)

        # Clean wind direction categories
        def clean_wind_direction(direction):
            direction = str(direction).lower().strip()
            direction_map = {
                'n': 'North', 'north': 'North', 'n.': 'North', 'northward': 'North',
                's': 'South', 'south': 'South', 's.': 'South', 'southward': 'South',
                'e': 'East', 'east': 'East', 'e.': 'East',
                'w': 'West', 'west': 'West', 'w.': 'West',
                'ne': 'North East', 'northeast': 'North East', 'ne.': 'North East',
                'nw': 'North West', 'northwest': 'North West', 'nw.': 'North West',
                'se': 'South East', 'southeast': 'South East', 'se.': 'South East',
                'sw': 'South West', 'southwest': 'South West', 'sw.': 'South West'
            }
            return direction_map.get(direction, np.nan)

        try:
            if 'Dew Point Category' in self.df.columns:
                self.df['Dew Point Category'] = self.df['Dew Point Category'].apply(clean_dew_point)

            if 'Wind Direction' in self.df.columns:
                self.df['Wind Direction'] = self.df['Wind Direction'].apply(clean_wind_direction)

        except KeyError as e:
            print(f"Key error: {e}")

        return self.df  

    # Cleans date column date
    def clean_date_data(self):
        
        if 'date' in self.df.columns: 
            try:
                self.df['date'] = pd.to_datetime(self.df['date'], format='%d/%m/%Y')
            except Exception as e:
                print(f"An error occurred while converting date: {e}")

        return self.df  
    # Removes outlier using IQR method
    def remove_outliers(self):
        print("Removing outliers...")
        for col in self.numeric_cols:
            if col not in rainfall_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Remove outliers
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
    # Impute numerical data using the mean
    def impute_numerical_data(self):
        for col in self.numeric_cols:
            # Fill missing values with the mean for numeric columns
            self.df[col].fillna(self.df[col].mean(), inplace=True)

            # Round the column values to 1 decimal place after filling missing values
            self.df[col] = self.df[col].round(1)

        return self.df  
    # Impute categorical data using mode
    def impute_string_data(self):
        for col in self.string_cols:
            # Fill missing values with the mode for string columns
            if not self.df[col].isnull().all(): 
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        return self.df  

    # Performs checks on df to see if missing values exist and categories are cleaned
    def check_data(self):
        print("Checking data...")
        all_numeric = True
        non_numeric_counts = {}

        # Check for non-numeric values in numeric columns
        for numeric_col in self.numeric_cols:
            non_numeric_values = self.df[numeric_col][self.df[numeric_col].isna() | self.df[numeric_col].isnull()]
            non_numeric_counts[numeric_col] = len(non_numeric_values)
            
            if non_numeric_counts[numeric_col] > 0:
                all_numeric = False
                print(f"Non-numeric values found in {numeric_col}: {non_numeric_counts[numeric_col]} occurrences")
                print(f"Sample of non-numeric values: {set(non_numeric_values.head())}")

        if all_numeric:
            print("No non-numeric values found in numeric columns.")
        else:
            print("Non-numeric values found in some numeric columns.")

        for string_col in self.string_cols:
            unique_strings = set(self.df[string_col].unique())
            print(f"Unique Strings in {string_col} after cleaning: {unique_strings}")
        print("********************************************************")
