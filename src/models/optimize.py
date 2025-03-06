"""
Model optimization module for credit risk assessment.
Functions for hyperparameter tuning and threshold optimization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Any, Union, Callable
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_hyperparameters(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series,
                           param_grid: Dict[str, List[Any]], cv: int = 5,
                           scoring: str = 'roc_auc', n_jobs: int = -1,
                           method: str = 'grid') -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Perform hyperparameter optimization
    
    Args:
        model: Base model to optimize
        X_train: Training features
        y_train: Training target
        param_grid: Parameter grid to search
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        method: Search method ('grid' or 'random')
        
    Returns:
        Tuple of (optimized model, best parameters)
    """
    logger.info(f"Optimizing hyperparameters using {method} search")
    
    if method == 'grid':
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
    elif method == 'random':
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=20,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Fit the search
    search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    return best_model, best_params

def optimize_probability_threshold(model: BaseEstimator, X_val: pd.DataFrame, y_val: pd.Series,
                                 metric: Union[str, Callable] = 'f1',
                                 thresholds: np.ndarray = None) -> Tuple[float, float]:
    """
    Optimize probability threshold for binary classification
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        metric: Metric to optimize ('f1', 'precision', 'recall', or custom function)
        thresholds: Array of thresholds to try
        
    Returns:
        Tuple of (optimal threshold, best metric value)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    logger.info(f"Optimizing probability threshold for {metric}")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Create thresholds if not provided
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    
    # Define metric function
    if metric == 'f1':
        metric_func = lambda y_true, y_pred: f1_score(y_true, y_pred)
    elif metric == 'precision':
        metric_func = lambda y_true, y_pred: precision_score(y_true, y_pred)
    elif metric == 'recall':
        metric_func = lambda y_true, y_pred: recall_score(y_true, y_pred)
    elif callable(metric):
        metric_func = metric
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Evaluate each threshold
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = metric_func(y_val, y_pred)
        results.append((threshold, score))
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results, columns=['threshold', 'score'])
    
    # Find the optimal threshold (maximizing the score)
    best_idx = results_df['score'].idxmax()
    optimal_threshold = results_df.loc[best_idx, 'threshold']
    best_score = results_df.loc[best_idx, 'score']
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}, {metric} score: {best_score:.4f}")
    
    # Plot threshold vs. metric
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['score'])
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
               label=f'Optimal threshold: {optimal_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel(f'{metric} score')
    plt.title(f'Threshold Optimization for {metric}')
    plt.legend()
    plt.grid(True)
    
    return optimal_threshold, best_score

def optimize_business_threshold(model: BaseEstimator, X_val: pd.DataFrame, y_val: pd.Series,
                             profit_matrix: Dict[str, float],
                             thresholds: np.ndarray = None) -> Tuple[float, float]:
    """
    Optimize threshold based on business metrics (profit/loss)
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        profit_matrix: Dictionary with profit/loss values for TP, FP, TN, FN
        thresholds: Array of thresholds to try
        
    Returns:
        Tuple of (optimal threshold, expected profit)
    """
    logger.info("Optimizing threshold based on business metrics")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Create thresholds if not provided
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    
    # Get profit values
    tp_profit = profit_matrix.get('true_positive', 0)
    fp_profit = profit_matrix.get('false_positive', 0)
    tn_profit = profit_matrix.get('true_negative', 0)
    fn_profit = profit_matrix.get('false_negative', 0)
    
    logger.info(f"Profit matrix: TP={tp_profit}, FP={fp_profit}, TN={tn_profit}, FN={fn_profit}")
    
    # Evaluate each threshold
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Count outcomes
        tp = np.sum((y_pred == 1) & (y_val == 1))
        fp = np.sum((y_pred == 1) & (y_val == 0))
        tn = np.sum((y_pred == 0) & (y_val == 0))
        fn = np.sum((y_pred == 0) & (y_val == 1))
        
        # Calculate total profit
        total_profit = (tp * tp_profit + fp * fp_profit + tn * tn_profit + fn * fn_profit)
        
        results.append((threshold, total_profit))
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results, columns=['threshold', 'profit'])
    
    # Get optimal threshold
    best_idx = results_df['profit'].idxmax()
    optimal_threshold = results_df.loc[best_idx, 'threshold']
    best_profit = results_df.loc[best_idx, 'profit']
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}, expected profit: {best_profit:.2f}")
    
    # Plot threshold vs. profit
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['profit'])
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
               label=f'Optimal threshold: {optimal_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Expected Profit')
    plt.title('Threshold Optimization for Business Metrics')
    plt.legend()
    plt.grid(True)
    
    return optimal_threshold, best_profit

def calibrate_model_probabilities(model: BaseEstimator, X_train: pd.DataFrame, 
                               y_train: pd.Series, method: str = 'isotonic') -> BaseEstimator:
    """
    Calibrate model probabilities
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        method: Calibration method ('isotonic' or 'sigmoid')
        
    Returns:
        Calibrated model
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    logger.info(f"Calibrating model probabilities using {method} method")
    
    # Create calibrated model
    calibrated_model = CalibratedClassifierCV(
        base_estimator=model,
        method=method,
        cv='prefit'
    )
    
    # Fit calibration
    calibrated_model.fit(X_train, y_train)
    
    return calibrated_model

def find_optimal_cutpoints(model: BaseEstimator, X_val: pd.DataFrame, y_val: pd.Series,
                         n_bins: int = 5) -> List[float]:
    """
    Find optimal cutpoints for risk tiers
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        n_bins: Number of risk tiers
        
    Returns:
        List of cutpoints
    """
    logger.info(f"Finding optimal cutpoints for {n_bins} risk tiers")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Find optimal cutpoints using decision tree
    from sklearn.tree import DecisionTreeClassifier
    
    # Create a 1D decision tree
    tree = DecisionTreeClassifier(max_depth=n_bins-1, random_state=42)
    
    # Reshape probabilities for fitting
    X_tree = y_pred_proba.reshape(-1, 1)
    
    # Fit the tree
    tree.fit(X_tree, y_val)
    
    # Get thresholds from the tree
    thresholds = tree.tree_.threshold
    
    # Filter out -2 (which indicates leaf nodes)
    thresholds = thresholds[thresholds != -2]
    
    # Sort thresholds
    thresholds = np.sort(thresholds)
    
    # Add 0 and 1 as boundaries
    cutpoints = [0] + thresholds.tolist() + [1]
    
    logger.info(f"Optimal cutpoints: {cutpoints}")
    
    return cutpoints

def optimize_model_ensemble_weights(models: List[BaseEstimator], X_val: pd.DataFrame, 
                                 y_val: pd.Series) -> List[float]:
    """
    Optimize weights for model ensemble
    
    Args:
        models: List of trained models
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        List of optimal weights
    """
    logger.info(f"Optimizing weights for ensemble of {len(models)} models")
    
    # Get predicted probabilities from each model
    probs = []
    for model in models:
        probs.append(model.predict_proba(X_val)[:, 1])
    
    # Stack probabilities
    X_opt = np.column_stack(probs)
    
    # Use logistic regression to find optimal weights
    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression(C=1.0, solver='lbfgs')
    lr.fit(X_opt, y_val)
    
    # Get weights
    weights = lr.coef_[0]
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    logger.info(f"Optimal weights: {weights}")
    
    return weights.tolist()