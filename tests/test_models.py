import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.models.price_prediction import PricePredictionModel
from src.models.churn_prediction import ChurnPredictionModel

class TestPricePredictionModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.current_time = "2025-05-17 19:29:59"
        self.current_user = "akshayaa-403"
        self.model = PricePredictionModel(
            experiment_name="test_price_prediction",
            current_time=self.current_time,
            current_user=self.current_user
        )
        
        # Create sample test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        self.y = np.random.normal(100, 20, 100)

    def test_prepare_data(self):
        """Test data preparation"""
        df = pd.DataFrame({
            'price': self.y,
            'feature1': self.X['feature1'],
            'feature2': self.X['feature2']
        })
        X_train, X_test, y_train, y_test = self.model.prepare_data(df)
        self.assertEqual(len(X_train) + len(X_test), len(df))

    def test_model_training(self):
        """Test model training"""
        self.model.initialize_models()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        results = self.model.train_and_evaluate(X_train, X_test, y_train, y_test)
        self.assertGreater(len(results), 0)

class TestChurnPredictionModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.current_time = "2025-05-17 19:29:59"
        self.current_user = "akshayaa-403"
        self.model = ChurnPredictionModel(
            experiment_name="test_churn_prediction",
            current_time=self.current_time,
            current_user=self.current_user
        )
        
        # Create sample test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        self.y = np.random.binomial(1, 0.3, 100)

    def test_prepare_data(self):
        """Test data preparation with SMOTE"""
        df = pd.DataFrame({
            'is_churned': self.y,
            'feature1': self.X['feature1'],
            'feature2': self.X['feature2']
        })
        X_train, X_test, y_train, y_test = self.model.prepare_data(df)
        self.assertEqual(len(X_train) + len(X_test), len(df))

    def test_model_training(self):
        """Test model training"""
        self.model.initialize_models()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y)
        results = self.model.train_and_evaluate(X_train, X_test, y_train, y_test)
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main()
