import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.evaluation import ModelEvaluator
from src.utils.visualization import ModelVisualizer

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.current_time = "2025-05-17 19:29:59"
        self.current_user = "akshayaa-403"
        self.evaluator = ModelEvaluator(self.current_time, self.current_user)
        
        # Create sample test data
        np.random.seed(42)
        self.y_true = np.random.normal(100, 20, 100)
        self.y_pred = self.y_true + np.random.normal(0, 10, 100)

    def test_evaluate_regression(self):
        """Test regression evaluation"""
        metrics = self.evaluator.evaluate_regression(
            self.y_true, self.y_pred, "TestModel"
        )
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)

    def test_evaluate_classification(self):
        """Test classification evaluation"""
        y_true = np.random.binomial(1, 0.3, 100)
        y_pred = np.random.binomial(1, 0.3, 100)
        y_pred_proba = np.random.uniform(0, 1, 100)
        
        metrics = self.evaluator.evaluate_classification(
            y_true, y_pred, y_pred_proba, "TestModel"
        )
        self.assertIn('roc_auc', metrics)
        self.assertIn('average_precision', metrics)

class TestModelVisualizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.current_time = "2025-05-17 19:29:59"
        self.current_user = "akshayaa-403"
        self.visualizer = ModelVisualizer(self.current_time, self.current_user)
        
        # Create sample test data
        np.random.seed(42)
        self.y_true = np.random.normal(100, 20, 100)
        self.y_pred = self.y_true + np.random.normal(0, 10, 100)

    def test_plot_regression_results(self):
        """Test regression plotting"""
        self.visualizer.plot_regression_results(
            self.y_true, self.y_pred, "TestModel"
        )

    def test_plot_feature_importance(self):
        """Test feature importance plotting"""
        feature_importance = pd.DataFrame({
            'feature': ['A', 'B', 'C'],
            'importance': [0.5, 0.3, 0.2]
        })
        self.visualizer.plot_feature_importance(
            feature_importance, "TestModel"
        )

if __name__ == '__main__':
    unittest.main()
