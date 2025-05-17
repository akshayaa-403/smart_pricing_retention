# Smart Pricing and Retention System for Airbnb

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Last Updated](https://img.shields.io/badge/Last%20Updated-2025--05--18-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Author](https://img.shields.io/badge/Author-akshayaa--403-orange)

A machine learning system for optimizing Airbnb listing prices and predicting customer churn using advanced analytics and ML techniques.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Testing](#testing)
- [Results and Metrics](#results-and-metrics)

## 🎯 Overview

This project implements a comprehensive system for:
1. **Price Prediction**: Optimize listing prices based on various features
2. **Churn Prediction**: Identify potential customer churn risks
3. **Feature Engineering**: Advanced feature creation from raw Airbnb data
4. **Model Evaluation**: Robust evaluation metrics and visualizations

## ✨ Features

### Data Processing
- Automated data cleaning and preprocessing
- Handling missing values and outliers
- Feature scaling and encoding
- Time-based feature generation

### Price Prediction
- Multiple model implementations (Random Forest, XGBoost, LightGBM)
- Hyperparameter optimization using Optuna
- Feature importance analysis
- Price range recommendations

### Churn Prediction
- Balanced class handling using SMOTE
- Multiple classification models
- Risk score calculation
- Early warning system

### Evaluation & Visualization
- Comprehensive metric reporting
- Interactive dashboards
- Feature importance plots
- Time series analysis

## 📁 Project Structure

```
smart_pricing_retention/
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── price_prediction.py
│   │   └── churn_prediction.py
│   └── utils/
│       ├── evaluation.py
│       └── visualization.py
├── config/
│   └── model_config.yaml
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   └── test_utils.py
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/akshayaa-403/smart_pricing_retention.git
cd smart_pricing_retention
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.price_prediction import PricePredictionModel
from src.models.churn_prediction import ChurnPredictionModel

# Initialize components
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()
price_model = PricePredictionModel()
churn_model = ChurnPredictionModel()

# Process data
listings_df, reviews_df, calendar_df, neighborhoods = preprocessor.preprocess_pipeline(data_paths)

# Engineer features
engineered_features = feature_engineer.engineer_features(
    listings_df, reviews_df, calendar_df
)

# Train price prediction model
price_results = price_model.train_and_evaluate(X_train, X_test, y_train, y_test)

# Train churn prediction model
churn_results = churn_model.train_and_evaluate(X_train, X_test, y_train, y_test)
```

### Advanced Usage

```python
# Use custom configuration
from yaml import safe_load

with open('config/model_config.yaml', 'r') as f:
    config = safe_load(f)

# Initialize with custom parameters
price_model = PricePredictionModel(
    experiment_name=config['price_prediction']['experiment_name']
)

# Optimize hyperparameters
best_params = price_model.optimize_hyperparameters(
    'XGBoost', X_train, y_train, n_trials=100
)

# Generate explanations
explanations = price_model.explain_predictions(X_test)
```

## 🔧 Model Details

### Price Prediction Models
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

Key metrics:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

### Churn Prediction Models
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier

Key metrics:
- ROC-AUC Score
- Precision-Recall AUC
- F1 Score
- Business Impact Metrics

## ⚙️ Configuration

The system is configured through `config/model_config.yaml`. Key sections:

```yaml
data:
  paths:
    listings: data/listings.csv
    reviews: data/reviews.csv
    calendar: data/calendar.csv

price_prediction:
  experiment_name: price_prediction
  models:
    RandomForest:
      n_estimators: 100
      max_depth: 15

churn_prediction:
  experiment_name: churn_prediction
  class_balance:
    method: SMOTE
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Individual test files:
```bash
python -m pytest tests/test_preprocessing.py
python -m pytest tests/test_feature_engineering.py
python -m pytest tests/test_models.py
python -m pytest tests/test_utils.py
```

## 📊 Results and Metrics

### Price Prediction Performance
- RMSE: ~$25-30
- R² Score: 0.85-0.89
- MAPE: 12-15%

### Churn Prediction Performance
- ROC-AUC: 0.82-0.85
- Precision: 0.78-0.82
- Recall: 0.75-0.80
