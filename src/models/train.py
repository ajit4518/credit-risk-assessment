"""
Model training module for credit risk assessment.
Functions for training different types of models.
"""
import pandas as pd
import numpy as np
import yaml
import joblib
import logging
import os
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_config(config_path: str = 'config/model_config.yaml') -> Dict[str, Any]:
    """
    Load model configuration
    
    Args:
        config_path: Path to model configuration file
        
    Returns:
        Dictionary with model configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                             model_config: Dict[str, Any]) -> BaseEstimator:
    """
    Train a logistic regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        model_config: Model configuration parameters
        
    Returns:
        Trained logistic regression model
    """
    from sklearn.linear_model import LogisticRegression
    
    logger.info("Training logistic regression model")
    
    # Get model parameters from config
    params = model_config.get('logistic_regression', {})
    
    # Create and train model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # Basic evaluation
    train_accuracy = model.score(X_train, y_train)
    logger.info(f"Logistic regression training accuracy: {train_accuracy:.4f}")
    
    return model

def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series,
                           model_config: Dict[str, Any],
                           cv: int = 5) -> BaseEstimator:
    """
    Train a gradient boosting model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target
        model_config: Model configuration parameters
        cv: Number of cross-validation folds
        
    Returns:
        Trained gradient boosting model
    """
    from sklearn.ensemble import GradientBoostingClassifier
    
    logger.info("Training gradient boosting model with hyperparameter tuning")
    
    # Get base parameters from config
    base_params = model_config.get('gradient_boosting', {})
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [base_params.get('n_estimators', 100), 
                         base_params.get('n_estimators', 100) + 100],
        'learning_rate': [base_params.get('learning_rate', 0.1), 
                         base_params.get('learning_rate', 0.1) / 2],
        'max_depth': [base_params.get('max_depth', 3), 
                      base_params.get('max_depth', 3) + 1]
    }
    
    # Create base model
    base_model = GradientBoostingClassifier(**base_params)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       model_config: Dict[str, Any]) -> BaseEstimator:
    """
    Train a random forest model
    
    Args:
        X_train: Training features
        y_train: Training target
        model_config: Model configuration parameters
        
    Returns:
        Trained random forest model
    """
    from sklearn.ensemble import RandomForestClassifier
    
    logger.info("Training random forest model")
    
    # Get model parameters from config
    params = model_config.get('random_forest', {})
    
    # Create and train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Feature importances
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 important features:")
    for i, row in feature_importances.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                 model_config: Dict[str, Any]) -> BaseEstimator:
    """
    Train an XGBoost model
    
    Args:
        X_train: Training features
        y_train: Training target
        model_config: Model configuration parameters
        
    Returns:
        Trained XGBoost model
    """
    try:
        import xgboost as xgb
        logger.info("Training XGBoost model")
        
        # Get model parameters from config
        params = model_config.get('xgboost', {})
        
        # Create and train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    except ImportError:
        logger.warning("XGBoost not installed. Falling back to gradient boosting.")
        return train_gradient_boosting(X_train, y_train, model_config)

def train_ensemble(X_train: pd.DataFrame, y_train: pd.Series,
                  model_config: Dict[str, Any],
                  models: Optional[List[BaseEstimator]] = None) -> BaseEstimator:
    """
    Train an ensemble of models
    
    Args:
        X_train: Training features
        y_train: Training target
        model_config: Model configuration parameters
        models: List of pre-trained models (optional)
        
    Returns:
        Trained ensemble model
    """
    from sklearn.ensemble import VotingClassifier
    
    logger.info("Training ensemble model")
    
    # If no models provided, train base models
    if models is None:
        logger.info("Training base models for ensemble")
        models = []
        
        # Logistic regression
        lr_model = train_logistic_regression(X_train, y_train, model_config)
        models.append(('lr', lr_model))
        
        # Gradient boosting
        gb_model = train_gradient_boosting(X_train, y_train, model_config)
        models.append(('gb', gb_model))
        
        # Random forest
        rf_model = train_random_forest(X_train, y_train, model_config)
        models.append(('rf', rf_model))
        
        try:
            # XGBoost if available
            xgb_model = train_xgboost(X_train, y_train, model_config)
            models.append(('xgb', xgb_model))
        except:
            logger.info("Skipping XGBoost in ensemble")
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft'  # Use predicted probabilities
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    return ensemble

def train_stacking_ensemble(X_train: pd.DataFrame, y_train: pd.Series,
                           model_config: Dict[str, Any]) -> BaseEstimator:
    """
    Train a stacking ensemble
    
    Args:
        X_train: Training features
        y_train: Training target
        model_config: Model configuration parameters
        
    Returns:
        Trained stacking ensemble
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    logger.info("Training stacking ensemble")
    
    # Create base models
    base_models = []
    
    # Logistic regression
    from sklearn.linear_model import LogisticRegression
    lr_params = model_config.get('logistic_regression', {})
    lr_model = LogisticRegression(**lr_params)
    base_models.append(('lr', lr_model))
    
    # Gradient boosting
    from sklearn.ensemble import GradientBoostingClassifier
    gb_params = model_config.get('gradient_boosting', {})
    gb_model = GradientBoostingClassifier(**gb_params)
    base_models.append(('gb', gb_model))
    
    # Random forest
    from sklearn.ensemble import RandomForestClassifier
    rf_params = model_config.get('random_forest', {})
    rf_model = RandomForestClassifier(**rf_params)
    base_models.append(('rf', rf_model))
    
    try:
        # XGBoost if available
        import xgboost as xgb
        xgb_params = model_config.get('xgboost', {})
        xgb_model = xgb.XGBClassifier(**xgb_params)
        base_models.append(('xgb', xgb_model))
    except ImportError:
        logger.info("XGBoost not available for stacking ensemble")
    
    # Create stacking ensemble
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # Train ensemble
    stacking.fit(X_train, y_train)
    
    return stacking

def save_model(model: BaseEstimator, model_path: str, model_name: str) -> str:
    """
    Save trained model to disk
    
    Args:
        model: Trained model to save
        model_path: Directory to save model
        model_name: Name of the model file
        
    Returns:
        Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save model
    full_path = os.path.join(model_path, f"{model_name}.pkl")
    joblib.dump(model, full_path)
    logger.info(f"Model saved to {full_path}")
    
    return full_path

def create_woe_scorecard(X_train: pd.DataFrame, y_train: pd.Series,
                        categorical_cols: List[str],
                        numerical_cols: List[str]) -> Dict[str, Any]:
    """
    Create a traditional WoE-based scorecard
    
    Args:
        X_train: Training features
        y_train: Training target
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        
    Returns:
        Dictionary with scorecard model and components
    """
    try:
        import scorecardpy as sc
        
        logger.info("Creating WoE-based scorecard")
        
        # Combine features and target
        df = X_train.copy()
        df['target'] = y_train
        
        # Select columns for binning
        cols_to_bin = categorical_cols + numerical_cols
        
        # Create WoE bins
        bins = sc.woebin(df, y="target", x=cols_to_bin)
        
        # Convert features to WoE values
        woe_df = sc.woebin_ply(df, bins)
        
        # Keep only the WoE columns and target
        woe_cols = [f"{col}_woe" for col in cols_to_bin]
        woe_df = woe_df[woe_cols + ['target']]
        
        # Train logistic regression on WoE values
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(class_weight='balanced', C=0.1)
        lr.fit(woe_df[woe_cols], woe_df['target'])
        
        # Generate scorecard
        card = sc.scorecard(bins, lr, cols_to_bin, points0=600, odds0=1/60, pdo=20)
        
        # Return components
        return {
            'bins': bins,
            'logistic_model': lr,
            'scorecard': card
        }
    
    except ImportError:
        logger.warning("scorecardpy not installed. Cannot create WoE scorecard.")
        return None

def create_model_pipeline(preprocessor: Any, model: BaseEstimator) -> Any:
    """
    Create a full pipeline combining preprocessing and model
    
    Args:
        preprocessor: Scikit-learn preprocessing pipeline
        model: Trained model
        
    Returns:
        Full pipeline
    """
    from sklearn.pipeline import Pipeline
    
    logger.info("Creating full model pipeline")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline