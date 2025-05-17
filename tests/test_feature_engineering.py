import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.feature_engineering import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.current_time = "2025-05-17 19:29:59"
        self.current_user = "akshayaa-403"
        self.engineer = FeatureEngineer(self.current_time, self.current_user)
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'price': [100, 200, 300],
            'accommodates': [2, 4, 6],
            'bathrooms': [1, 2, 3],
            'bedrooms': [1, 2, 3],
            'latitude': [40.7, 40.8, 40.9],
            'longitude': [-74.0, -74.1, -74.2]
        })

    def test_create_price_features(self):
        """Test price feature creation"""
        result = self.engineer.create_price_features(self.sample_data.copy())
        self.assertIn('price_per_person', result.columns)
        self.assertIn('price_per_bedroom', result.columns)

    def test_create_location_features(self):
        """Test location feature creation"""
        result = self.engineer.create_location_features(self.sample_data.copy())
        self.assertIn('distance_to_center', result.columns)

    def test_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline"""
        listings = self.sample_data.copy()
        reviews = pd.DataFrame({
            'listing_id': [1, 1, 2],
            'date': ['2025-01-01', '2025-01-02', '2025-01-01'],
            'reviewer_id': [1, 2, 3]
        })
        calendar = pd.DataFrame({
            'listing_id': [1, 1, 2],
            'date': ['2025-01-01', '2025-01-02', '2025-01-01'],
            'available': ['t', 'f', 't'],
            'price': ['$100', '$200', '$300']
        })
        
        result = self.engineer.engineer_features(listings, reviews, calendar)
        self.assertGreater(len(result.columns), len(listings.columns))

if __name__ == '__main__':
    unittest.main()
