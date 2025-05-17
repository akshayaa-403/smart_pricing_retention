import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.current_time = "2025-05-17 19:29:59"
        self.current_user = "akshayaa-403"
        self.preprocessor = DataPreprocessor(self.current_time, self.current_user)
        
        # Create sample test data
        self.sample_listings = pd.DataFrame({
            'price': ['$100', '$200', '$300', None],
            'accommodates': [2, 4, None, 2],
            'bathrooms': [1.0, 2.0, 1.5, None],
            'bedrooms': [1, 2, None, 1]
        })

    def test_clean_price_columns(self):
        """Test price column cleaning"""
        cleaned_df = self.preprocessor.clean_price_columns(self.sample_listings.copy())
        self.assertTrue(pd.api.types.is_float_dtype(cleaned_df['price']))
        self.assertEqual(cleaned_df['price'].iloc[0], 100.0)

    def test_handle_missing_values(self):
        """Test missing value handling"""
        cleaned_df = self.preprocessor.handle_missing_values(self.sample_listings.copy())
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)

    def test_validate_data(self):
        """Test data validation"""
        valid_df = pd.DataFrame({
            'price': [100, 200],
            'accommodates': [2, 4],
            'bathrooms': [1.0, 2.0],
            'bedrooms': [1, 2]
        })
        self.assertTrue(self.preprocessor.validate_data(valid_df, 'listings'))

if __name__ == '__main__':
    unittest.main()
