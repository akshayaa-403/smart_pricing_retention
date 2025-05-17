import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
from joblib import Memory, Parallel, delayed
from datetime import datetime

# Set up caching
memory = Memory(location='./cache', verbose=0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='feature_engineering.log'
)

class FeatureEngineer:
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['price_per_person'] = df['price'] / df['accommodates'].replace(0, 1)
            df['price_per_bedroom'] = df['price'] / df['bedrooms'].replace(0, 1)
            df['price_per_bathroom'] = df['price'] / df['bathrooms'].replace(0, 1)
            df['price_per_bed'] = df['price'] / df['beds'].replace(0, 1)
            
            # Calculate price ratios and percentiles
            df['weekly_price_ratio'] = df['weekly_price'] / (df['price'] * 7)
            df['monthly_price_ratio'] = df['monthly_price'] / (df['price'] * 30)
            df['price_percentile'] = df['price'].rank(pct=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error creating price features: {str(e)}")
            raise

    @memory.cache
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Calculate distance from city center (example coordinates for NYC)
            center_lat, center_lon = 40.7128, -74.0060
            
            df['distance_to_center'] = np.sqrt(
                (df['latitude'] - center_lat)**2 + 
                (df['longitude'] - center_lon)**2
            )
            
            # Create location clusters (if many listings)
            if len(df) > 1000:
                from sklearn.cluster import KMeans
                coords = df[['latitude', 'longitude']].values
                kmeans = KMeans(n_clusters=10, random_state=42)
                df['location_cluster'] = kmeans.fit_predict(coords)
            
            return df
            
        except Exception as e:
            logging.error(f"Error creating location features: {str(e)}")
            raise

    def create_amenity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Create essential amenities features
            essential_amenities = [
                'Wifi', 'Air conditioning', 'Kitchen', 'Heating', 'Washer',
                'Coffee maker', 'TV', 'Microwave', 'Refrigerator'
            ]
            
            for amenity in essential_amenities:
                df[f'has_{amenity.lower().replace(" ", "_")}'] = df['amenities'].str.contains(
                    amenity, case=False, regex=False, na=False
                ).astype(int)
            
            # Calculate amenity scores
            df['amenity_score'] = df[[f'has_{am.lower().replace(" ", "_")}' 
                                    for am in essential_amenities]].sum(axis=1)
            
            df['amenity_ratio'] = df['amenity_score'] / len(essential_amenities)
            
            return df
            
        except Exception as e:
            logging.error(f"Error creating amenity features: {str(e)}")
            raise

    def create_review_features(self, df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Calculate review statistics
            review_stats = reviews_df.groupby('listing_id').agg({
                'reviewer_id': 'count',
                'date': ['min', 'max']
            }).reset_index()
            
            review_stats.columns = [
                'listing_id', 'review_count', 
                'first_review_date', 'last_review_date'
            ]
            
            # Calculate review frequency
            review_stats['days_between_reviews'] = (
                pd.to_datetime(review_stats['last_review_date']) - 
                pd.to_datetime(review_stats['first_review_date'])
            ).dt.days
            
            review_stats['review_frequency'] = (
                review_stats['review_count'] / 
                review_stats['days_between_reviews'].replace(0, 1)
            )
            
            return df.merge(review_stats, on='listing_id', how='left')
            
        except Exception as e:
            logging.error(f"Error creating review features: {str(e)}")
            raise

    def create_booking_features(self, df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Calculate availability statistics
            booking_stats = calendar_df.groupby('listing_id').agg({
                'available': lambda x: (x == 't').mean(),
                'price': ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            booking_stats.columns = [
                'listing_id', 'availability_rate',
                'avg_price', 'price_std', 'min_price', 'max_price'
            ]
            
            # Calculate price volatility
            booking_stats['price_volatility'] = (
                booking_stats['price_std'] / booking_stats['avg_price']
            ).fillna(0)
            
            return df.merge(booking_stats, on='listing_id', how='left')
            
        except Exception as e:
            logging.error(f"Error creating booking features: {str(e)}")
            raise

    def engineer_features(self, 
                        listings_df: pd.DataFrame, 
                        reviews_df: pd.DataFrame, 
                        calendar_df: pd.DataFrame) -> pd.DataFrame:
                          
        try:
            logging.info("Starting feature engineering pipeline...")
            
            # Create all features
            df = listings_df.copy()
            df = self.create_price_features(df)
            df = self.create_location_features(df)
            df = self.create_amenity_features(df)
            df = self.create_review_features(df, reviews_df)
            df = self.create_booking_features(df, calendar_df)
            
            logging.info("Feature engineering completed successfully")
            return df
            
        except Exception as e:
            logging.error(f"Error in feature engineering pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    # Load your preprocessed data here
    # engineered_df = engineer.engineer_features(listings_df, reviews_df, calendar_df)
