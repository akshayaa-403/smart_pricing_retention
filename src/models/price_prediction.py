import pandas as pd
import numpy as np
import logging
import mlflow
import optuna
import shap
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='price_prediction.log'
)

class PricePredictionModel:
    def __init__(self, 
                 experiment_name: str = "price_prediction",
                 current_time: str = "2025-05-18 00:48:05",
                 current_user: str = "akshayaa-403"):
        self.current_time = current_time
        self.current_user = current_user
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('-inf')
        self.feature_importance = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        logging.info(f"Initialized PricePredictionModel at {self.current_time} by {self.current_user}")

    def prepare_data(self, 
                    df: pd.DataFrame, 
                    target_col: str = 'price',
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            # Remove outliers
            Q1 = df[target_col].quantile(0.01)
            Q3 = df[target_col].quantile(0.99)
            IQR = Q3 - Q1
            df = df[
                (df[target_col] >= Q1 - 1.5 * IQR) & 
                (df[target_col] <= Q3 + 1.5 * IQR)
            ]

            # Log transform target
            y = np.log1p(df[target_col])
            X = df.drop(columns=[target_col])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            logging.info(f"Data prepared: Training set size: {X_train.shape}, Test set size: {X_test.shape}")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error in data preparation: {str(e)}")
            raise

    def initialize_models(self, custom_params: Optional[Dict] = None) -> None:
        try:
            default_params = {
                'RandomForest': {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                'XGBoost': {
                    'n_estimators': 100,
                    'max_depth': 7,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'LightGBM': {
                    'n_estimators': 100,
                    'max_depth': 7,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }

            params = custom_params if custom_params else default_params

            self.models = {
                'RandomForest': RandomForestRegressor(**params['RandomForest']),
                'XGBoost': XGBRegressor(**params['XGBoost']),
                'LightGBM': LGBMRegressor(**params['LightGBM'])
            }

            logging.info("Models initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise

    def optimize_hyperparameters(self, 
                               model_name: str, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               n_trials: int = 100) -> Dict:
        try:
            def objective(trial):
                if model_name == 'RandomForest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                    }
                    model = RandomForestRegressor(**params, random_state=42)
                
                elif model_name == 'XGBoost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
                    }
                    model = XGBRegressor(**params, random_state=42)
                
                elif model_name == 'LightGBM':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
                    }
                    model = LGBMRegressor(**params, random_state=42)

                score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
                return score

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            logging.info(f"Hyperparameter optimization completed for {model_name}")
            return study.best_params

        except Exception as e:
            logging.error(f"Error in hyperparameter optimization: {str(e)}")
            raise

    def train_and_evaluate(self, 
                          X_train: pd.DataFrame, 
                          X_test: pd.DataFrame, 
                          y_train: pd.Series, 
                          y_test: pd.Series) -> Dict:
        try:
            results = {}
            
            with mlflow.start_run(run_name=f"training_run_{self.current_time}"):
                for name, model in self.models.items():
                    logging.info(f"Training {name}...")
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Store results
                    results[name] = {
                        'model': model,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    }
                    
                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        f'{name}_rmse': rmse,
                        f'{name}_mae': mae,
                        f'{name}_r2': r2
                    })
                    
                    # Update best model
                    if r2 > self.best_score:
                        self.best_score = r2
                        self.best_model = model
                        self.best_model_name = name
                    
                    # Calculate feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = model.feature_importances_

            logging.info("Model training and evaluation completed")
            return results

        except Exception as e:
            logging.error(f"Error in model training and evaluation: {str(e)}")
            raise

    def explain_predictions(self, 
                          X: pd.DataFrame, 
                          n_samples: int = 100) -> Dict:
        try:
            if self.best_model is None:
                raise ValueError("No best model available. Train models first.")

            # Calculate SHAP values
            explainer = shap.TreeExplainer(self.best_model)
            shap_values = explainer.shap_values(X.sample(n_samples, random_state=42))

            return {
                'shap_values': shap_values,
                'feature_names': X.columns
            }

        except Exception as e:
            logging.error(f"Error generating SHAP explanations: {str(e)}")
            raise

    def save_model(self, 
                  filename: str = 'best_price_prediction_model.joblib') -> None:
        try:
            if self.best_model is None:
                raise ValueError("No best model to save. Train models first.")

            model_data = {
                'model': self.best_model,
                'metadata': {
                    'model_name': self.best_model_name,
                    'timestamp': self.current_time,
                    'user': self.current_user,
                    'score': self.best_score,
                    'feature_importance': self.feature_importance.get(self.best_model_name)
                }
            }

            joblib.dump(model_data, filename)
            logging.info(f"Model saved successfully as {filename}")

        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize model
        price_model = PricePredictionModel()

        # Load your data here
        df = pd.read_csv('engineered_features.csv')

        # Prepare data
        X_train, X_test, y_train, y_test = price_model.prepare_data(df)

        # Initialize and train models
        price_model.initialize_models()
        results = price_model.train_and_evaluate(X_train, X_test, y_train, y_test)

        # Optimize best model
        best_params = price_model.optimize_hyperparameters(price_model.best_model_name, X_train, y_train)

        # Generate explanations
        explanations = price_model.explain_predictions(X_test)

        # Save model
        price_model.save_model()

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
