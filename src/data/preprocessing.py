import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from datetime import datetime
import geopandas as gpd
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='preprocessing.log'
)

class DataPreprocessor:
    def load_data(self, data_paths: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
        try:
            logging.info("Loading datasets...")
            listings_df = pd.read_csv(data_paths['listings'], low_memory=False)
            reviews_df = pd.read_csv(data_paths['reviews'], low_memory=False)
            calendar_df = pd.read_csv(data_paths['calendar'], low_memory=False)
            neighborhoods = gpd.read_file(data_paths['neighborhoods'])
            
            return listings_df, reviews_df, calendar_df, neighborhoods
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame, dataset_name: str) -> bool:
        try:
            if dataset_name == 'listings':
                required_columns = ['price', 'accommodates', 'bathrooms', 'bedrooms']
                assert all(col in df.columns for col in required_columns), "Missing required columns"
                assert df['price'].min() >= 0, "Invalid price values"
                assert df['accommodates'].min() > 0, "Invalid accommodates values"
            
            logging.info(f"Data validation passed for {dataset_name}")
            return True
            
        except AssertionError as e:
            logging.error(f"Validation failed for {dataset_name}: {str(e)}")
            raise

    def clean_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        price_columns = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee']
        
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].replace({r'\$': '', ',': ''}, regex=True).astype(float)
        
        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        # Numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median() if strategy == 'median' else df[col].mode()[0])

        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract basic time components
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add seasons
            df['season'] = df['date'].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            # Add holiday season flag (approximate)
            df['is_holiday_season'] = df['date'].dt.month.isin([11, 12]).astype(int)

        return df

    def preprocess_pipeline(self, data_paths: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
        try:
            # Load data
            listings_df, reviews_df, calendar_df, neighborhoods = self.load_data(data_paths)
            
            # Validate data
            for df, name in zip([listings_df, reviews_df, calendar_df], 
                              ['listings', 'reviews', 'calendar']):
                self.validate_data(df, name)
            
            # Clean price columns
            listings_df = self.clean_price_columns(listings_df)
            calendar_df = self.clean_price_columns(calendar_df)
            
            # Handle missing values
            listings_df = self.handle_missing_values(listings_df)
            reviews_df = self.handle_missing_values(reviews_df)
            calendar_df = self.handle_missing_values(calendar_df)
            
            # Create time features
            calendar_df = self.create_time_features(calendar_df)
            reviews_df = self.create_time_features(reviews_df)
            
            logging.info("Preprocessing pipeline completed successfully")
            return listings_df, reviews_df, calendar_df, neighborhoods
            
        except Exception as e:
            logging.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    data_paths = {
        'listings': 'data/listings.csv',
        'reviews': 'data/reviews.csv',
        'calendar': 'data/calendar.csv',
        'neighborhoods': 'data/neighbourhoods.geojson'
    }
    
    preprocessor = DataPreprocessor()
    listings_df, reviews_df, calendar_df, neighborhoods = preprocessor.preprocess_pipeline(data_paths)
