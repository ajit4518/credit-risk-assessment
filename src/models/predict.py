"""
Model prediction module for credit risk assessment.
Functions for making predictions with trained models.
"""
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Any, Union
from sklearn.base import BaseEstimator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path: str) -> BaseEstimator:
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model

def predict_default_probability(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """
    Predict default probabilities for new data
    
    Args:
        model: Trained model
        X: Feature data
        
    Returns:
        Array of default probabilities
    """
    logger.info(f"Predicting default probabilities for {len(X)} samples")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    return y_pred_proba

def predict_default_with_threshold(model: BaseEstimator, X: pd.DataFrame, 
                                 threshold: float = 0.5) -> np.ndarray:
    """
    Predict default classification with specified threshold
    
    Args:
        model: Trained model
        X: Feature data
        threshold: Probability threshold for classification
        
    Returns:
        Array of binary default predictions
    """
    logger.info(f"Predicting default with threshold {threshold} for {len(X)} samples")
    
    # Get predicted probabilities
    y_pred_proba = predict_default_probability(model, X)
    
    # Apply threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred

def calculate_risk_tiers(default_probs: np.ndarray, tiers: int = 5) -> np.ndarray:
    """
    Assign risk tiers based on default probabilities
    
    Args:
        default_probs: Array of default probabilities
        tiers: Number of risk tiers
        
    Returns:
        Array of risk tier assignments
    """
    logger.info(f"Calculating {tiers} risk tiers for {len(default_probs)} samples")
    
    # Define tier boundaries (using percentiles)
    tier_boundaries = np.linspace(0, 1, tiers + 1)
    
    # Assign tiers
    risk_tiers = np.zeros(len(default_probs))
    for i in range(tiers):
        lower = tier_boundaries[i]
        upper = tier_boundaries[i + 1]
        
        if i == tiers - 1:  # Include upper bound for last tier
            mask = (default_probs >= lower) & (default_probs <= upper)
        else:
            mask = (default_probs >= lower) & (default_probs < upper)
        
        risk_tiers[mask] = i + 1
    
    return risk_tiers

def calculate_risk_score(default_probs: np.ndarray, score_min: int = 300,
                        score_max: int = 850) -> np.ndarray:
    """
    Calculate credit risk scores from default probabilities
    
    Args:
        default_probs: Array of default probabilities
        score_min: Minimum score value
        score_max: Maximum score value
        
    Returns:
        Array of risk scores
    """
    logger.info(f"Calculating risk scores [{score_min}-{score_max}] for {len(default_probs)} samples")
    
    # Invert default probability (higher probability means lower score)
    inverse_prob = 1 - default_probs
    
    # Scale to desired range
    scores = score_min + inverse_prob * (score_max - score_min)
    
    # Round to integers
    scores = np.round(scores).astype(int)
    
    return scores

def generate_model_explanations(model: BaseEstimator, X: pd.DataFrame, 
                              row_index: Union[int, List[int]]) -> Dict[str, Any]:
    """
    Generate explanations for model predictions
    
    Args:
        model: Trained model
        X: Feature data
        row_index: Index of sample(s) to explain
        
    Returns:
        Dictionary with explanation information
    """
    try:
        import shap
        
        logger.info(f"Generating model explanations")
        
        # Extract sample(s) to explain
        if isinstance(row_index, int):
            X_explain = X.iloc[[row_index]]
        else:
            X_explain = X.iloc[row_index]
        
        # Create explainer based on model type
        model_type = type(model).__name__
        
        if "GradientBoosting" in model_type or "XGB" in model_type:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)
        
        # Calculate SHAP values
        shap_values = explainer(X_explain)
        
        # Get feature importance
        if isinstance(row_index, int):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values.values[0])
            }).sort_values('importance', ascending=False)
            
            # Get top features and their impact
            top_features = feature_importance.head(10)
            
            explanations = {
                'top_features': top_features.to_dict(orient='records'),
                'shap_values': shap_values.values.tolist(),
                'expected_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
            }
        else:
            # Average absolute SHAP values across multiple samples
            avg_impact = np.abs(shap_values.values).mean(axis=0)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': avg_impact
            }).sort_values('importance', ascending=False)
            
            # Get top features and their impact
            top_features = feature_importance.head(10)
            
            explanations = {
                'top_features': top_features.to_dict(orient='records'),
                'shap_values': shap_values.values.tolist(),
                'expected_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
            }
        
        return explanations
    
    except ImportError:
        logger.warning("SHAP not installed. Cannot generate explanations.")
        
        # Fallback to simpler explanations using permutation importance
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X, np.zeros(len(X)), n_repeats=5, random_state=42)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        
        # Get top features
        top_features = feature_importance.head(10)
        
        explanations = {
            'top_features': top_features.to_dict(orient='records'),
            'method': 'permutation_importance'
        }
        
        return explanations

def apply_scorecard(scorecard: Dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a WoE scorecard to calculate credit scores
    
    Args:
        scorecard: Scorecard dictionary (from create_woe_scorecard)
        X: Feature data
        
    Returns:
        DataFrame with credit scores
    """
    import scorecardpy as sc
    
    logger.info(f"Applying scorecard to {len(X)} samples")
    
    # Apply WoE transformation
    woe_df = sc.woebin_ply(X, scorecard['bins'])
    
    # Calculate scores
    scores = sc.scorecard_ply(X, scorecard['scorecard'])
    
    return scores

def perform_what_if_analysis(model: BaseEstimator, X: pd.DataFrame, 
                           row_index: int, feature_changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform what-if analysis for a specific sample
    
    Args:
        model: Trained model
        X: Feature data
        row_index: Index of sample to analyze
        feature_changes: Dictionary of feature changes to simulate
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Performing what-if analysis for sample {row_index}")
    
    # Get original sample
    X_orig = X.iloc[[row_index]].copy()
    
    # Get original prediction
    orig_prob = predict_default_probability(model, X_orig)[0]
    
    # Create modified sample
    X_mod = X_orig.copy()
    
    # Apply changes
    for feature, new_value in feature_changes.items():
        if feature in X_mod.columns:
            X_mod[feature] = new_value
        else:
            logger.warning(f"Feature {feature} not found in data")
    
    # Get new prediction
    new_prob = predict_default_probability(model, X_mod)[0]
    
    # Calculate impact
    impact = new_prob - orig_prob
    
    # Return results
    results = {
        'original_features': X_orig.iloc[0].to_dict(),
        'modified_features': X_mod.iloc[0].to_dict(),
        'original_probability': orig_prob,
        'new_probability': new_prob,
        'impact': impact,
        'percentage_change': (impact / orig_prob) * 100 if orig_prob > 0 else float('inf')
    }
    
    return results

def batch_score_portfolio(model: BaseEstimator, X: pd.DataFrame, 
                         threshold: float = 0.5) -> pd.DataFrame:
    """
    Score an entire portfolio of accounts
    
    Args:
        model: Trained model
        X: Feature data
        threshold: Probability threshold for classification
        
    Returns:
        DataFrame with prediction results
    """
    logger.info(f"Batch scoring portfolio with {len(X)} accounts")
    
    # Create results DataFrame
    results = pd.DataFrame()
    
    # If loan_id exists, add it to results
    if 'loan_id' in X.columns:
        results['loan_id'] = X['loan_id']
    
    if 'customer_id' in X.columns:
        results['customer_id'] = X['customer_id']
    
    # Generate predictions
    probabilities = predict_default_probability(model, X)
    
    # Add predictions to results
    results['default_probability'] = probabilities
    results['default_prediction'] = (probabilities >= threshold).astype(int)
    
    # Calculate risk score
    results['risk_score'] = calculate_risk_score(probabilities)
    
    # Assign risk tiers
    tier_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    results['risk_tier'] = pd.cut(
        probabilities, 
        bins=[0, 0.05, 0.1, 0.2, 0.3, 1.0],
        labels=tier_labels
    )
    
    return results