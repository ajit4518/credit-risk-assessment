"""
Model evaluation module for credit risk assessment.
Functions for evaluating model performance and fairness.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Any, Union
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix, 
    classification_report, average_precision_score, roc_curve
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_classification_metrics(model: BaseEstimator, X_test: pd.DataFrame, 
                                  y_test: pd.Series, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate classification metrics for a model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        threshold: Probability threshold for classification
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, average_precision_score
    )
    
    logger.info(f"Evaluating model with threshold {threshold}")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Print metrics
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics

def plot_roc_curve(model: BaseEstimator, X_test: pd.DataFrame, 
                  y_test: pd.Series, save_path: str = None) -> plt.Figure:
    """
    Plot ROC curve for a model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting ROC curve")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Add labels and legend
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"ROC curve saved to {save_path}")
    
    return fig

def plot_precision_recall_curve(model: BaseEstimator, X_test: pd.DataFrame, 
                              y_test: pd.Series, save_path: str = None) -> plt.Figure:
    """
    Plot precision-recall curve for a model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting precision-recall curve")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, color='blue', lw=2, 
            label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    
    # Add labels and legend
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Precision-recall curve saved to {save_path}")
    
    return fig

def plot_confusion_matrix(model: BaseEstimator, X_test: pd.DataFrame, 
                         y_test: pd.Series, threshold: float = 0.5,
                         save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix for a model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        threshold: Probability threshold for classification
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    logger.info(f"Plotting confusion matrix with threshold {threshold}")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
    disp.plot(cmap='Blues', ax=ax)
    
    # Add title
    ax.set_title('Confusion Matrix')
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig

def plot_score_distribution(model: BaseEstimator, X_test: pd.DataFrame, 
                           y_test: pd.Series, save_path: str = None) -> plt.Figure:
    """
    Plot distribution of model scores by actual outcome
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting score distribution")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot distributions
    sns.histplot(y_pred_proba[y_test == 0], bins=50, alpha=0.5, color='blue', 
                 label='Non-Default', ax=ax)
    sns.histplot(y_pred_proba[y_test == 1], bins=50, alpha=0.5, color='red', 
                 label='Default', ax=ax)
    
    # Add labels and legend
    ax.set_xlabel('Predicted Default Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Default Probability Scores')
    ax.legend()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Score distribution saved to {save_path}")
    
    return fig

def plot_calibration_curve(model: BaseEstimator, X_test: pd.DataFrame, 
                          y_test: pd.Series, n_bins: int = 10,
                          save_path: str = None) -> plt.Figure:
    """
    Plot calibration curve for a model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        n_bins: Number of bins for calibration
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    from sklearn.calibration import calibration_curve
    
    logger.info("Plotting calibration curve")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
    
    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    
    # Add labels and legend
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve')
    ax.legend(loc='lower right')
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Calibration curve saved to {save_path}")
    
    return fig

def calculate_ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        
    Returns:
        KS statistic value
    """
    # Sort scores and corresponding true values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Count of positive and negative examples
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    # Cumulative sums
    cum_pos = np.cumsum(y_true)
    cum_neg = np.cumsum(1 - y_true)
    
    # True positive and false positive rates
    tpr = cum_pos / n_pos
    fpr = cum_neg / n_neg
    
    # Calculate KS statistic
    ks_values = np.abs(tpr - fpr)
    ks_statistic = np.max(ks_values)
    
    return ks_statistic

def plot_ks_curve(model: BaseEstimator, X_test: pd.DataFrame, 
                 y_test: pd.Series, save_path: str = None) -> plt.Figure:
    """
    Plot Kolmogorov-Smirnov curve
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting KS curve")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate KS statistic
    ks_stat = calculate_ks_statistic(y_test.values, y_pred_proba)
    
    # Sort data by predicted probabilities
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    sorted_y_test = y_test.values[sorted_indices]
    sorted_proba = y_pred_proba[sorted_indices]
    
    # Calculate cumulative distributions
    n_pos = np.sum(sorted_y_test)
    n_neg = len(sorted_y_test) - n_pos
    
    cum_pos = np.cumsum(sorted_y_test) / n_pos
    cum_neg = np.cumsum(1 - sorted_y_test) / n_neg
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot cumulative distributions
    ax.plot(np.arange(len(sorted_y_test)) / len(sorted_y_test), cum_pos, 
            label='Cumulative Default Rate', color='red')
    ax.plot(np.arange(len(sorted_y_test)) / len(sorted_y_test), cum_neg, 
            label='Cumulative Non-Default Rate', color='blue')
    
    # Find maximum KS value
    ks_idx = np.argmax(np.abs(cum_pos - cum_neg))
    ks_x = ks_idx / len(sorted_y_test)
    
    # Plot KS line
    ax.plot([ks_x, ks_x], [cum_neg[ks_idx], cum_pos[ks_idx]], 'k--', 
            label=f'KS = {ks_stat:.4f}')
    
    # Add labels and legend
    ax.set_xlabel('Percentage of population')
    ax.set_ylabel('Cumulative rate')
    ax.set_title('Kolmogorov-Smirnov Curve')
    ax.legend()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"KS curve saved to {save_path}")
    
    return fig

def analyze_model_fairness(model: BaseEstimator, X_test: pd.DataFrame, 
                         y_test: pd.Series, protected_cols: List[str],
                         save_path: str = None) -> Dict[str, Dict[str, float]]:
    """
    Analyze model fairness across protected attributes
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        protected_cols: List of protected attribute columns
        save_path: Path to save the results
        
    Returns:
        Dictionary with fairness metrics by protected attribute
    """
    logger.info("Analyzing model fairness")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Results dictionary
    fairness_metrics = {}
    
    # Analyze each protected attribute
    for col in protected_cols:
        if col not in X_test.columns:
            logger.warning(f"Protected attribute {col} not found in data")
            continue
        
        logger.info(f"Analyzing fairness for attribute: {col}")
        
        # Get unique values
        unique_values = X_test[col].unique()
        
        # Calculate metrics for each group
        group_metrics = {}
        
        for value in unique_values:
            # Filter data for this group
            mask = (X_test[col] == value)
            group_y_test = y_test[mask]
            group_y_pred = y_pred[mask]
            group_y_pred_proba = y_pred_proba[mask]
            
            # Skip if too few samples
            if len(group_y_test) < 10:
                logger.warning(f"Too few samples for {col}={value}, skipping")
                continue
            
            # Calculate metrics
            group_metrics[value] = {
                'count': len(group_y_test),
                'default_rate': group_y_test.mean(),
                'approval_rate': 1 - group_y_pred.mean(),  # Approval = not predicted default
                'avg_score': group_y_pred_proba.mean(),
                'auc': roc_auc_score(group_y_test, group_y_pred_proba) if len(np.unique(group_y_test)) > 1 else np.nan
            }
        
        # Add to results
        fairness_metrics[col] = group_metrics
    
    # Save results if path provided
    if save_path:
        # Convert to DataFrame
        fairness_df = pd.DataFrame()
        for col, metrics in fairness_metrics.items():
            for value, value_metrics in metrics.items():
                row = pd.DataFrame({
                    'attribute': col,
                    'value': value,
                    **value_metrics
                }, index=[0])
                fairness_df = pd.concat([fairness_df, row], ignore_index=True)
        
        # Save to CSV
        fairness_df.to_csv(save_path, index=False)
        logger.info(f"Fairness metrics saved to {save_path}")
    
    return fairness_metrics

def calculate_population_stability_index(expected: np.ndarray, 
                                        actual: np.ndarray, 
                                        buckets: int = 10) -> float:
    """
    Calculate Population Stability Index
    
    Args:
        expected: Expected distribution (e.g., training scores)
        actual: Actual distribution (e.g., test scores)
        buckets: Number of buckets for binning
        
    Returns:
        PSI value
    """
    logger.info(f"Calculating PSI with {buckets} buckets")
    
    # Define buckets based on expected distribution
    breaks = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    
    # Ensure unique breaks (can happen with discrete data)
    breaks = np.unique(breaks)
    
    # Count observations in each bucket
    expected_counts = np.histogram(expected, breaks)[0]
    actual_counts = np.histogram(actual, breaks)[0]
    
    # Calculate percentages
    expected_pcts = expected_counts / expected_counts.sum()
    actual_pcts = actual_counts / actual_counts.sum()
    
    # Replace zeros with small number to avoid division by zero
    expected_pcts = np.where(expected_pcts == 0, 0.0001, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, 0.0001, actual_pcts)
    
    # Calculate PSI
    psi_values = (actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)
    psi = np.sum(psi_values)
    
    logger.info(f"Population Stability Index: {psi:.4f}")
    logger.info(f"PSI Interpretation: {'High shift (>0.25)' if psi > 0.25 else 'Medium shift (0.1-0.25)' if psi > 0.1 else 'Low shift (<0.1)'}")
    
    return psi