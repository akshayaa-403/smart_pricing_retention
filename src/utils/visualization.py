import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='visualization.log'
)

class ModelVisualizer:
    def __init__(self, current_time: str = "2025-05-18 00:59:09", current_user: str = "akshayaa-403"):
        self.current_time = current_time
        self.current_user = current_user
        # Set default style
        plt.style.use('seaborn')
        logging.info(f"Initialized ModelVisualizer at {self.current_time} by {self.current_user}")

    def plot_regression_results(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              model_name: str,
                              save_path: Optional[str] = None) -> None:

        try:
            fig = plt.figure(figsize=(15, 5))

            # Scatter plot
            plt.subplot(121)
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{model_name} Predictions vs Actual')

            # Residuals plot
            plt.subplot(122)
            residuals = y_pred - y_true
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                logging.info(f"Regression plot saved to {save_path}")
            plt.close()

        except Exception as e:
            logging.error(f"Error in regression plotting: {str(e)}")
            raise

    def plot_classification_results(self,
                                  metrics: Dict,
                                  model_name: str,
                                  save_path: Optional[str] = None) -> None:

        try:
            fig = plt.figure(figsize=(15, 5))

            # ROC curve
            plt.subplot(131)
            plt.plot(metrics['roc_curve']['fpr'], 
                    metrics['roc_curve']['tpr'], 
                    label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()

            # Precision-Recall curve
            plt.subplot(132)
            plt.plot(metrics['pr_curve']['recall'],
                    metrics['pr_curve']['precision'],
                    label=f'PR curve (AP = {metrics["average_precision"]:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()

            # Confusion Matrix
            plt.subplot(133)
            sns.heatmap(metrics['confusion_matrix'],
                       annot=True,
                       fmt='d',
                       cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                logging.info(f"Classification plot saved to {save_path}")
            plt.close()

        except Exception as e:
            logging.error(f"Error in classification plotting: {str(e)}")
            raise

    def plot_feature_importance(self,
                              feature_importance: pd.DataFrame,
                              model_name: str,
                              save_path: Optional[str] = None) -> None:

        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(
                x='importance',
                y='feature',
                data=feature_importance,
                palette='viridis'
            )
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                logging.info(f"Feature importance plot saved to {save_path}")
            plt.close()

        except Exception as e:
            logging.error(f"Error in feature importance plotting: {str(e)}")
            raise

    def plot_predictions_over_time(self,
                                 dates: pd.Series,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str,
                                 save_path: Optional[str] = None) -> None:

        try:
            plt.figure(figsize=(15, 6))
            plt.plot(dates, y_true, label='Actual', alpha=0.7)
            plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title(f'{model_name} - Predictions Over Time')
            plt.legend()
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                logging.info(f"Time series plot saved to {save_path}")
            plt.close()

        except Exception as e:
            logging.error(f"Error in time series plotting: {str(e)}")
            raise

    def create_interactive_dashboard(self,
                                  metrics: Dict,
                                  feature_importance: pd.DataFrame,
                                  save_path: Optional[str] = None) -> None:

        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Feature Importance',
                    'ROC Curve',
                    'Precision-Recall Curve',
                    'Confusion Matrix'
                )
            )

            # Feature Importance
            fig.add_trace(
                go.Bar(
                    x=feature_importance['importance'],
                    y=feature_importance['feature'],
                    orientation='h'
                ),
                row=1, col=1
            )

            # ROC Curve
            fig.add_trace(
                go.Scatter(
                    x=metrics['roc_curve']['fpr'],
                    y=metrics['roc_curve']['tpr'],
                    name=f'ROC (AUC={metrics["roc_auc"]:.2f})'
                ),
                row=1, col=2
            )

            # Precision-Recall Curve
            fig.add_trace(
                go.Scatter(
                    x=metrics['pr_curve']['recall'],
                    y=metrics['pr_curve']['precision'],
                    name=f'PR (AP={metrics["average_precision"]:.2f})'
                ),
                row=2, col=1
            )

            # Confusion Matrix
            fig.add_trace(
                go.Heatmap(
                    z=metrics['confusion_matrix'],
                    x=['Predicted 0', 'Predicted 1'],
                    y=['Actual 0', 'Actual 1'],
                    colorscale='Blues'
                ),
                row=2, col=2
            )

            fig.update_layout(height=800, width=1200, title_text="Model Performance Dashboard")
            
            if save_path:
                fig.write_html(save_path)
                logging.info(f"Interactive dashboard saved to {save_path}")

        except Exception as e:
            logging.error(f"Error creating interactive dashboard: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        visualizer = ModelVisualizer()
        
        # Example regression visualization
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        y_true = np.random.normal(100, 20, 100)
        y_pred = y_true + np.random.normal(0, 10, 100)
        visualizer.plot_predictions_over_time(dates, y_true, y_pred, "RandomForest")
        
        # Example feature importance visualization
        feature_importance = pd.DataFrame({
            'feature': ['A', 'B', 'C', 'D'],
            'importance': [0.4, 0.3, 0.2, 0.1]
        })
        visualizer.plot_feature_importance(feature_importance, "XGBoost")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
