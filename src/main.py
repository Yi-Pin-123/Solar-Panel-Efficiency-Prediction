from data_ingestion import DataIngestion
from data_preprocessing import Cleaning
from model import SolarEfficiencyModel
from constants import *
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():

    weather_ingestor  = DataIngestion('data/weather.db')
    air_quality_ingestor = DataIngestion('data/air_quality.db')

    # Load weather and air quality data
    weather_df = weather_ingestor.get_df()
    air_quality_df = air_quality_ingestor .get_df()

    # Create cleaner objects which cleans the df
    weather_cleaner = Cleaning(weather_df, weather_numeric_columns, weather_string_columns)
    air_quality_cleaner = Cleaning(air_quality_df, air_quality_numeric_columns, air_quality_string_columns)

    # Merge datasets on 'data_ref' and 'date'
    df = pd.merge(weather_cleaner.df, air_quality_cleaner.df, on=['data_ref', 'date'])

    # Run three models
    model = SolarEfficiencyModel(df)
    model.run()

if __name__ == '__main__':
    main()
