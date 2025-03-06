"""
Model analysis visualizations for credit risk assessment.
Functions for visualizing model performance and explanations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_learning_curve(model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                      cv: int = 5, train_sizes: np.ndarray = None,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot learning curve for a model
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data
        cv: Number of cross-validation folds
        train_sizes: Array of training sizes
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    from sklearn.model_selection import learning_curve
    
    logger.info("Plotting learning curve")
    
    # Define train sizes if not provided
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='roc_auc',
        n_jobs=-1, random_state=42
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot learning curve
    ax.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Training score')
    ax.plot(train_sizes, test_mean, 'o-', color='#e74c3c', label='Cross-validation score')
    
    # Add confidence intervals
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                   alpha=0.1, color='#3498db')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                   alpha=0.1, color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score (ROC AUC)')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_shap_summary(model: BaseEstimator, X: pd.DataFrame,
                    max_display: int = 20,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot SHAP summary for model feature importance
    
    Args:
        model: Trained model
        X: Feature data
        max_display: Maximum number of features to display
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    try:
        import shap
        
        logger.info("Plotting SHAP summary")
        
        # Determine model type and create explainer
        model_type = type(model).__name__
        
        if hasattr(model, "predict_proba"):
            # For most tree ensemble models
            if "GradientBoosting" in model_type or "XGB" in model_type or "RandomForest" in model_type:
                explainer = shap.TreeExplainer(model)
            # For most other models
            else:
                explainer = shap.Explainer(model)
            
            # Calculate SHAP values
            shap_values = explainer(X)
            
            # Create figure
            plt.figure(figsize=(10, 12))
            
            # Plot summary
            if hasattr(shap_values, "values"):
                # For newer SHAP versions
                shap.summary_plot(shap_values.values, X, max_display=max_display, show=False)
            else:
                # For older SHAP versions
                shap.summary_plot(shap_values, X, max_display=max_display, show=False)
            
            # Get the current figure
            fig = plt.gcf()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            return fig
        else:
            logger.warning("Model does not support predict_proba. Cannot create SHAP plot.")
            return None
    
    except ImportError:
        logger.warning("SHAP not installed. Cannot create SHAP plot.")
        return None

def plot_feature_importance_permutation(model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                                      n_repeats: int = 10, random_state: int = 42,
                                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance using permutation importance
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data
        n_repeats: Number of permutation repeats
        random_state: Random seed
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    from sklearn.inspection import permutation_importance
    
    logger.info("Plotting permutation feature importance")
    
    # Calculate permutation importance
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    
    # Sort features by importance
    sorted_idx = result.importances_mean.argsort()[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot importance
    sns.barplot(
        x=result.importances_mean[sorted_idx],
        y=[X.columns[i] for i in sorted_idx],
        xerr=result.importances_std[sorted_idx],
        ax=ax
    )
    
    # Add labels and title
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Mean Decrease in Accuracy")
    ax.set_ylabel("Feature")
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_model_comparison(models: List[Tuple[str, BaseEstimator]], 
                       X: pd.DataFrame, y: pd.Series,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for multiple models
    
    Args:
        models: List of (name, model) tuples
        X: Feature data
        y: Target data
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    logger.info("Plotting model comparison")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each model
    for name, model in models:
        # Get predicted probabilities
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            
            # Calculate AUC
            auc = roc_auc_score(y, y_pred_proba)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
        else:
            logger.warning(f"Model {name} does not support predict_proba")
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    # Add labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_partial_dependence(model: BaseEstimator, X: pd.DataFrame,
                         features: List[Union[str, int, Tuple[str, str]]],
                         save_path: Optional[str] = None) -> List[plt.Figure]:
    """
    Plot partial dependence plots for selected features
    
    Args:
        model: Trained model
        X: Feature data
        features: List of features to plot (can be column names, indices, or tuples for 2D plots)
        save_path: Base path to save plots (will append feature names)
        
    Returns:
        List of Matplotlib figures
    """
    from sklearn.inspection import partial_dependence, plot_partial_dependence
    
    logger.info("Plotting partial dependence")
    
    # Create a list to store figures
    figures = []
    
    # Convert feature names to indices if necessary
    feature_indices = []
    feature_names = []
    
    for feature in features:
        if isinstance(feature, str):
            # Single feature by name
            if feature in X.columns:
                idx = list(X.columns).index(feature)
                feature_indices.append(idx)
                feature_names.append(feature)
            else:
                logger.warning(f"Feature '{feature}' not found in data")
        elif isinstance(feature, int):
            # Single feature by index
            if 0 <= feature < len(X.columns):
                feature_indices.append(feature)
                feature_names.append(X.columns[feature])
            else:
                logger.warning(f"Feature index {feature} out of range")
        elif isinstance(feature, tuple) and len(feature) == 2:
            # Feature pair for 2D plot
            f1, f2 = feature
            if f1 in X.columns and f2 in X.columns:
                idx1 = list(X.columns).index(f1)
                idx2 = list(X.columns).index(f2)
                feature_indices.append((idx1, idx2))
                feature_names.append(f"{f1}_vs_{f2}")
            else:
                logger.warning(f"Feature pair '{f1}' and '{f2}' not found in data")
        else:
            logger.warning(f"Invalid feature specification: {feature}")
    
    # Plot each feature or feature pair
    for i, (idx, name) in enumerate(zip(feature_indices, feature_names)):
        if isinstance(idx, tuple):
            # 2D plot for feature pair
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate partial dependence
            pdp = partial_dependence(model, X, features=[idx])
            
            # Get results
            XX, YY = np.meshgrid(pdp["values"][0], pdp["values"][1])
            Z = pdp["average"][0].T
            
            # Create contour plot
            CS = ax.contour(XX, YY, Z, levels=10, cmap="viridis", linewidths=0.5)
            ax.clabel(CS, inline=True, fontsize=10)
            
            # Add colored contour
            cax = ax.contourf(XX, YY, Z, levels=10, cmap="viridis", alpha=0.5)
            fig.colorbar(cax)
            
            # Add feature names
            f1_name = X.columns[idx[0]]
            f2_name = X.columns[idx[1]]
            
            ax.set_xlabel(f1_name)
            ax.set_ylabel(f2_name)
            ax.set_title(f'Partial Dependence: {f1_name} vs {f2_name}')
        else:
            # 1D plot for single feature
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate partial dependence
            pdp = partial_dependence(model, X, features=[idx])
            
            # Get results
            x_values = pdp["values"][0]
            y_values = pdp["average"][0]
            
            # Plot partial dependence
            ax.plot(x_values, y_values, 'b-', linewidth=2)
            
            # Add grid and feature name
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(X.columns[idx])
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'Partial Dependence for {X.columns[idx]}')
        
        # Save if path provided
        if save_path:
            save_file = f"{save_path}_{name}.png" if save_path else None
            if save_file:
                plt.savefig(save_file)
                logger.info(f"Plot saved to {save_file}")
        
        figures.append(fig)
    
    return figures

def plot_calibration_curve(models: List[Tuple[str, BaseEstimator]], 
                        X: pd.DataFrame, y: pd.Series,
                        n_bins: int = 10,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot calibration curves for multiple models
    
    Args:
        models: List of (name, model) tuples
        X: Feature data
        y: Target data
        n_bins: Number of bins for calibration
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    from sklearn.calibration import calibration_curve
    
    logger.info("Plotting calibration curves")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot calibration curve for each model
    for name, model in models:
        # Get predicted probabilities
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=n_bins)
            
            # Plot calibration curve
            ax.plot(prob_pred, prob_true, 's-', label=name)
        else:
            logger.warning(f"Model {name} does not support predict_proba")
    
    # Plot ideal calibration
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Add labels and title
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curves')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_cumulative_gains(model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot cumulative gains chart
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting cumulative gains chart")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Sort by predicted probability (descending)
    indices = np.argsort(y_pred_proba)[::-1]
    y_sorted = y.iloc[indices].values
    
    # Calculate cumulative gains
    y_cum = np.cumsum(y_sorted)
    cum_gain = y_cum / y_cum[-1]
    
    # Calculate percentages
    percentages = np.arange(1, len(y) + 1) / len(y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot cumulative gains
    ax.plot(percentages, cum_gain, 'b-', linewidth=2, label='Model')
    
    # Plot random model (diagonal line)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    # Add labels and title
    ax.set_xlabel('Percentage of sample')
    ax.set_ylabel('Percentage of positive outcomes')
    ax.set_title('Cumulative Gains Chart')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_lift_curve(model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                  n_bins: int = 10, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot lift curve
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data
        n_bins: Number of bins for lift calculation
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting lift curve")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'prob': y_pred_proba,
        'target': y.values
    })
    
    # Sort by probability (descending)
    df = df.sort_values('prob', ascending=False)
    
    # Create deciles (or other bins)
    df['bin'] = pd.qcut(df.index, n_bins, labels=False)
    
    # Calculate overall default rate
    overall_rate = df['target'].mean()
    
    # Calculate default rate by bin
    bin_stats = df.groupby('bin').agg(
        count=('target', 'count'),
        defaults=('target', 'sum'),
        default_rate=('target', 'mean')
    )
    
    # Calculate lift
    bin_stats['lift'] = bin_stats['default_rate'] / overall_rate
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot lift curve
    x = np.arange(n_bins) + 1
    width = 0.4
    
    # Bar chart for lift
    bars = ax.bar(x, bin_stats['lift'], width, color='#3498db', label='Lift')
    
    # Add lift values above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height:.2f}x', ha='center', va='bottom')
    
    # Add default rate line on secondary Y-axis
    ax2 = ax.twinx()
    ax2.plot(x, bin_stats['default_rate'] * 100, 'ro-', label='Default Rate')
    
    # Add baseline
    ax.axhline(y=1, color='grey', linestyle='--')
    
    # Add labels and title
    ax.set_xlabel('Bin (Sorted by Predicted Probability)')
    ax.set_ylabel('Lift')
    ax2.set_ylabel('Default Rate (%)')
    ax.set_title('Lift Curve')
    
    # Set x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels([f'{100 - i * 100/n_bins:.0f}-{100 - (i+1) * 100/n_bins:.0f}%' for i in range(n_bins)])
    plt.xticks(rotation=45)
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig