from data_merger import DataMerger  # Import the DataMerger class

# Define database paths, table names, and merge column
db1_path = "./data/weather.db"  # Replace with the actual path
db2_path = "./data/air_quality.db"  # Replace with the actual path
table1_name = "weather"  # Replace with the actual table name
table2_name = "air_quality"  # Replace with the actual table name
merge_column = ["data_ref", "date"]  # Replace with the actual column name

merger = DataMerger(db1_path, db2_path, table1_name, table2_name, merge_column)
merged_data = merger.merge_remove_duplicate()


from data_cleaner import DataCleaning  # Import the DataCleaner class


# Initialize the DataCleaning class
cleaner = DataCleaning(input_dataframe=merged_data)

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

effiency_order = ['Low', 'Moderate', 'High']

    # Convert Daily Solar Panel Efficiency column to an ordinal variable
cleaner.convert_to_ordinal('Daily Solar Panel Efficiency', effiency_order)

# Save category metadata
category_metadata = {
    'Dew Point Category': dew_point_order,
    'Daily Solar Panel Efficiency': effiency_order
}

# Get and display the cleaned DataFrame
cleaned_df = cleaner.get_dataframe()

from build_model import ModelRunner  # Import the ModelRunner class
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# Convert ordinal categorical columns to their numerical codes
cleaned_df = cleaned_df.apply(lambda col: col.cat.codes if col.dtype.name == 'category' else col)

cleaned_df['month'] = cleaned_df['date'].dt.month

X = cleaned_df.drop(['Daily Solar Panel Efficiency',  
                        'date', 
                        'data_ref', 
                        'psi_north', 'psi_south', 'psi_east','psi_west', 'psi_central', 
                        'pm25_north', 'pm25_south', 'pm25_east','pm25_west',
                        'Wet Bulb Temperature (deg F)', 'Air Pressure (hPa)', 'Dew Point Category'
                        ], axis=1)

y = cleaned_df['Daily Solar Panel Efficiency']

# Assuming X and y are already defined and preprocessed
categorical_columns = ['Wind Direction']  # Replace with your actual categorical columns

# Instantiate the ModelRunner
runner = ModelRunner(X, y, categorical_columns)

# Define models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    runner.train_and_evaluate(model, model_name)
    print(f"Train Results for {model_name}:")
    print(runner.get_results()[model_name])


