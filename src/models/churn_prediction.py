import pandas as pd
import numpy as np
import logging
import mlflow
import optuna
import shap
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='churn_prediction.log'
)

class ChurnPredictionModel:
    def __init__(self, 
                 experiment_name: str = "churn_prediction",
                 current_time: str = "2025-05-18 00:50:00",
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
        logging.info(f"Initialized ChurnPredictionModel at {self.current_time} by {self.current_user}")

    def prepare_data(self, 
                    df: pd.DataFrame,
                    target_col: str = 'is_churned',
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            # Split features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Apply SMOTE to balance classes
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            logging.info(f"Data prepared: Training set size: {X_train_balanced.shape}, Test set size: {X_test.shape}")
            return X_train_balanced, X_test, y_train_balanced, y_test

        except Exception as e:
            logging.error(f"Error in data preparation: {str(e)}")
            raise

    def initialize_models(self, custom_params: Optional[Dict] = None) -> None:
        try:
            default_params = {
                'RandomForest': {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'class_weight': 'balanced',
                    'random_state': 42
                },
                'XGBoost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'scale_pos_weight': 1,
                    'random_state': 42
                },
                'LightGBM': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }

            params = custom_params if custom_params else default_params

            self.models = {
                'RandomForest': RandomForestClassifier(**params['RandomForest']),
                'XGBoost': XGBClassifier(**params['XGBoost']),
                'LightGBM': LGBMClassifier(**params['LightGBM'])
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
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
                        'class_weight': 'balanced'
                    }
                    model = RandomForestClassifier(**params, random_state=42)
                
                elif model_name == 'XGBoost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0)
                    }
                    model = XGBClassifier(**params, random_state=42)
                
                elif model_name == 'LightGBM':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 50)
                    }
                    model = LGBMClassifier(**params, random_state=42)

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                score = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
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
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    auc_roc = roc_auc_score(y_test, y_pred_proba)
                    avg_precision = average_precision_score(y_test, y_pred_proba)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred)
                    
                    # Store results
                    results[name] = {
                        'model': model,
                        'auc_roc': auc_roc,
                        'avg_precision': avg_precision,
                        'confusion_matrix': conf_matrix,
                        'classification_report': class_report
                    }
                    
                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        f'{name}_auc_roc': auc_roc,
                        f'{name}_avg_precision': avg_precision
                    })
                    
                    # Update best model
                    if auc_roc > self.best_score:
                        self.best_score = auc_roc
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
                  filename: str = 'best_churn_prediction_model.joblib') -> None:
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
        churn_model = ChurnPredictionModel()

        # Load your data here
        df = pd.read_csv('engineered_features.csv')

        # Prepare data
        X_train, X_test, y_train, y_test = churn_model.prepare_data(df)

        # Initialize and train models
        churn_model.initialize_models()
        results = churn_model.train_and_evaluate(X_train, X_test, y_train, y_test)

        # Optimize best model
        best_params = churn_model.optimize_hyperparameters(churn_model.best_model_name, X_train, y_train)

        # Generate explanations
        explanations = churn_model.explain_predictions(X_test)

        # Save model
        churn_model.save_model()

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
