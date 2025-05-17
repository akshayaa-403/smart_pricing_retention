import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='evaluation.log'
)

class ModelEvaluator:
    def __init__(self, current_time: str = "2025-05-17 19:21:09", current_user: str = "akshayaa-403"):
        self.current_time = current_time
        self.current_user = current_user
        logging.info(f"Initialized ModelEvaluator at {self.current_time} by {self.current_user}")

    def evaluate_regression(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          model_name: str) -> Dict:
                            
        try:
            # Transform back from log scale if needed
            if np.all(y_true > 0) and np.all(y_pred > 0):
                y_true_orig = np.expm1(y_true)
                y_pred_orig = np.expm1(y_pred)
            else:
                y_true_orig = y_true
                y_pred_orig = y_pred

            # Calculate metrics
            mse = mean_squared_error(y_true_orig, y_pred_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_orig, y_pred_orig)
            r2 = r2_score(y_true_orig, y_pred_orig)
            
            # Calculate MAPE
            mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
            
            # Calculate custom metrics
            within_10_percent = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig) <= 0.1) * 100
            within_20_percent = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig) <= 0.2) * 100

            metrics = {
                'model_name': model_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'within_10_percent': within_10_percent,
                'within_20_percent': within_20_percent
            }

            logging.info(f"Regression evaluation completed for {model_name}")
            return metrics

        except Exception as e:
            logging.error(f"Error in regression evaluation: {str(e)}")
            raise

    def evaluate_classification(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_pred_proba: np.ndarray,
                              model_name: str) -> Dict:
                                
        try:
            # Calculate basic metrics
            class_report = classification_report(y_true, y_pred, output_dict=True)
            conf_mat = confusion_matrix(y_true, y_pred)
            
            # Calculate ROC and PR curves
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            # Calculate custom business metrics
            true_positives = conf_mat[1][1]
            false_positives = conf_mat[0][1]
            false_negatives = conf_mat[1][0]
            
            # Cost of false positives (e.g., unnecessary intervention)
            cost_fp = 100  # Example cost
            # Cost of false negatives (e.g., lost customer)
            cost_fn = 500  # Example cost
            
            total_cost = (false_positives * cost_fp) + (false_negatives * cost_fn)

            metrics = {
                'model_name': model_name,
                'classification_report': class_report,
                'confusion_matrix': conf_mat,
                'roc_auc': roc_auc,
                'average_precision': avg_precision,
                'roc_curve': {
                    'fpr': fpr,
                    'tpr': tpr
                },
                'pr_curve': {
                    'precision': precision,
                    'recall': recall
                },
                'business_metrics': {
                    'total_cost': total_cost,
                    'cost_per_prediction': total_cost / len(y_true)
                }
            }

            logging.info(f"Classification evaluation completed for {model_name}")
            return metrics

        except Exception as e:
            logging.error(f"Error in classification evaluation: {str(e)}")
            raise

    def compare_models(self, 
                      model_results: List[Dict], 
                      task_type: str = 'regression') -> pd.DataFrame:
                        
        try:
            comparison_df = pd.DataFrame()
            
            if task_type == 'regression':
                metrics = ['rmse', 'mae', 'r2', 'mape']
                for result in model_results:
                    model_metrics = {
                        metric: result[metric] 
                        for metric in metrics
                    }
                    comparison_df = comparison_df.append(
                        {
                            'model': result['model_name'],
                            **model_metrics
                        },
                        ignore_index=True
                    )
            
            elif task_type == 'classification':
                metrics = ['roc_auc', 'average_precision']
                for result in model_results:
                    model_metrics = {
                        metric: result[metric] 
                        for metric in metrics
                    }
                    comparison_df = comparison_df.append(
                        {
                            'model': result['model_name'],
                            **model_metrics,
                            'f1_score': result['classification_report']['weighted avg']['f1-score']
                        },
                        ignore_index=True
                    )

            logging.info("Model comparison completed")
            return comparison_df

        except Exception as e:
            logging.error(f"Error in model comparison: {str(e)}")
            raise

    def calculate_feature_importance(self, 
                                   model: object, 
                                   feature_names: List[str],
                                   top_n: int = 20) -> pd.DataFrame:
                                     
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                raise AttributeError("Model doesn't have feature importance attributes")

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })

            feature_importance = feature_importance.sort_values(
                'importance', ascending=False
            ).head(top_n)

            logging.info("Feature importance calculation completed")
            return feature_importance

        except Exception as e:
            logging.error(f"Error in feature importance calculation: {str(e)}")
            raise

    def generate_evaluation_report(self, 
                                 metrics: Dict,
                                 task_type: str,
                                 output_path: Optional[str] = None) -> str:

        try:
            report = f"Model Evaluation Report\n"
            report += f"Generated at: {self.current_time}\n"
            report += f"Generated by: {self.current_user}\n"
            report += f"{'='*50}\n\n"

            if task_type == 'regression':
                report += f"Regression Metrics:\n"
                report += f"RMSE: {metrics['rmse']:.4f}\n"
                report += f"MAE: {metrics['mae']:.4f}\n"
                report += f"R²: {metrics['r2']:.4f}\n"
                report += f"MAPE: {metrics['mape']:.2f}%\n"
                report += f"Within 10%: {metrics['within_10_percent']:.2f}%\n"
                report += f"Within 20%: {metrics['within_20_percent']:.2f}%\n"
            
            else:  # classification
                report += f"Classification Metrics:\n"
                report += f"ROC AUC: {metrics['roc_auc']:.4f}\n"
                report += f"Average Precision: {metrics['average_precision']:.4f}\n"
                report += f"\nClassification Report:\n"
                report += f"{metrics['classification_report']}\n"
                report += f"\nBusiness Impact:\n"
                report += f"Total Cost: ${metrics['business_metrics']['total_cost']:,.2f}\n"
                report += f"Cost per Prediction: ${metrics['business_metrics']['cost_per_prediction']:.2f}\n"

            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report)
                logging.info(f"Evaluation report saved to {output_path}")

            return report

        except Exception as e:
            logging.error(f"Error generating evaluation report: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        evaluator = ModelEvaluator()
        
        # Example regression evaluation
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        regression_metrics = evaluator.evaluate_regression(y_true, y_pred, "RandomForest")
        
        # Example classification evaluation
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_pred_proba = np.array([0.2, 0.8, 0.9, 0.6, 0.3])
        classification_metrics = evaluator.evaluate_classification(
          y_true, y_pred, y_pred_proba, "XGBoost"
        )

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
