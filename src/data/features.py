"""
Feature engineering module for credit risk assessment.
Functions for creating, selecting, and transforming features.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
from typing import Dict, List, Tuple, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_payment_features(data: pd.DataFrame, payment_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from payment history
    
    Args:
        data: Main DataFrame
        payment_data: Payment history DataFrame
        
    Returns:
        DataFrame with additional payment features
    """
    logger.info("Creating payment behavior features")
    result = data.copy()
    
    # Group payment data by loan ID
    if 'loan_id' not in payment_data.columns:
        logger.error("loan_id column missing from payment_data")
        return result
    
    # Aggregate payment features
    payment_features = payment_data.groupby('loan_id').agg(
        avg_payment_ratio=('payment_amount', lambda x: np.mean(x / payment_data.loc[x.index, 'scheduled_payment_amount'])),
        max_days_late=('days_late', 'max'),
        pct_late_payments=('days_late', lambda x: np.mean(x > 0)),
        num_missed_payments=('days_late', lambda x: np.sum(x > 30)),
        payment_volatility=('payment_amount', 'std')
    )
    
    # Merge with main data
    result = result.merge(payment_features, on='loan_id', how='left')
    
    # Fill missing values
    for col in payment_features.columns:
        if result[col].isnull().sum() > 0:
            logger.info(f"Filling {result[col].isnull().sum()} missing values in {col}")
            result[col] = result[col].fillna(0)
    
    return result

def create_time_features(data: pd.DataFrame, date_column: str = 'origination_date') -> pd.DataFrame:
    """
    Create features from date column
    
    Args:
        data: DataFrame to process
        date_column: Name of the date column to process
        
    Returns:
        DataFrame with additional time-based features
    """
    logger.info(f"Creating time-based features from {date_column}")
    result = data.copy()
    
    # Convert to datetime if needed
    if pd.api.types.is_string_dtype(result[date_column]):
        result[date_column] = pd.to_datetime(result[date_column])
    
    # Extract date components
    result[f'{date_column}_year'] = result[date_column].dt.year
    result[f'{date_column}_month'] = result[date_column].dt.month
    result[f'{date_column}_quarter'] = result[date_column].dt.quarter
    
    # Calculate loan age in months (time since origination)
    result['loan_age_months'] = (pd.Timestamp('today') - result[date_column]).dt.days / 30
    
    return result

def create_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features
    
    Args:
        data: DataFrame to process
        
    Returns:
        DataFrame with additional interaction features
    """
    logger.info("Creating interaction features")
    result = data.copy()
    
    # Payment-to-income ratio
    if 'scheduled_payment_amount' in result.columns and 'income' in result.columns:
        result['payment_income_ratio'] = result['scheduled_payment_amount'] / result['income'].clip(lower=1)
    
    # Debt-to-income × utilization
    if 'debt_to_income_ratio' in result.columns and 'utilization_ratio' in result.columns:
        result['dti_utilization'] = result['debt_to_income_ratio'] * result['utilization_ratio']
    
    # Loan amount to income
    if 'loan_amount' in result.columns and 'income' in result.columns:
        result['loan_to_income'] = result['loan_amount'] / result['income'].clip(lower=1)
    
    # Credit score × debt-to-income interaction (higher credit might offset higher DTI)
    if 'credit_score' in result.columns and 'debt_to_income_ratio' in result.columns:
        # Normalize credit score to [0,1] range
        credit_norm = (result['credit_score'] - 300) / 550  # Assuming 300-850 range
        result['credit_dti_interaction'] = credit_norm * (1 - result['debt_to_income_ratio'])
    
    return result

def create_risk_flags(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary risk flag features
    
    Args:
        data: DataFrame to process
        
    Returns:
        DataFrame with additional risk flag features
    """
    logger.info("Creating risk flag features")
    result = data.copy()
    
    # High DTI flag
    if 'debt_to_income_ratio' in result.columns:
        result['high_dti_flag'] = (result['debt_to_income_ratio'] > 0.43).astype(int)
    
    # Low credit score flag
    if 'credit_score' in result.columns:
        result['low_credit_flag'] = (result['credit_score'] < 620).astype(int)
    
    # Payment history flags
    if 'max_days_late' in result.columns:
        result['serious_delinquency_flag'] = (result['max_days_late'] > 60).astype(int)
    
    if 'num_missed_payments' in result.columns:
        result['multiple_missed_payments_flag'] = (result['num_missed_payments'] > 2).astype(int)
    
    if 'avg_payment_ratio' in result.columns:
        result['underpayment_flag'] = (result['avg_payment_ratio'] < 0.9).astype(int)
    
    # Combined high risk flag
    risk_flags = [col for col in result.columns if col.endswith('_flag')]
    if risk_flags:
        result['high_risk_flag'] = (result[risk_flags].sum(axis=1) > 0).astype(int)
    
    return result

def select_features(X: pd.DataFrame, y: pd.Series, 
                   method: str = 'f_classif',
                   k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select most important features
    
    Args:
        X: Feature DataFrame
        y: Target Series
        method: Feature selection method ('f_classif', 'mutual_info', 'model_based')
        k: Number of features to select
        
    Returns:
        Tuple of (selected feature DataFrame, list of selected feature names)
    """
    logger.info(f"Selecting top {k} features using {method}")
    
    if method == 'f_classif':
        selector = SelectKBest(f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == 'model_based':
        # Use a tree-based model to select features
        from sklearn.ensemble import RandomForestClassifier
        
        # Train a random forest
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        forest.fit(X, y)
        
        # Get feature importances
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        
        # Select top k features
        selected_features = [X.columns[i] for i in indices]
        logger.info(f"Selected features: {selected_features}")
        
        return X[selected_features], selected_features
    else:
        logger.error(f"Unknown feature selection method: {method}")
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Apply feature selection
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    mask = selector.get_support()
    selected_features = X.columns[mask].tolist()
    
    logger.info(f"Selected features: {selected_features}")
    
    return X[selected_features], selected_features

def create_polynomial_features(X: pd.DataFrame, 
                              degree: int = 2,
                              interaction_only: bool = True) -> pd.DataFrame:
    """
    Create polynomial features
    
    Args:
        X: Feature DataFrame
        degree: Polynomial degree
        interaction_only: Whether to include only interaction terms
        
    Returns:
        DataFrame with polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    logger.info(f"Creating polynomial features with degree={degree}, interaction_only={interaction_only}")
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(X.columns)
    
    # Create DataFrame with new features
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    # Remove original features (they're duplicated)
    for col in X.columns:
        if col in X_poly_df.columns:
            X_poly_df = X_poly_df.drop(columns=[col])
    
    logger.info(f"Created {X_poly_df.shape[1]} polynomial features")
    
    # Combine original and polynomial features
    result = pd.concat([X, X_poly_df], axis=1)
    
    return result

def add_macroeconomic_features(data: pd.DataFrame, 
                              macro_data: pd.DataFrame,
                              date_column: str = 'origination_date') -> pd.DataFrame:
    """
    Add macroeconomic indicators to the dataset based on date
    
    Args:
        data: Main DataFrame
        macro_data: Macroeconomic data with date index
        date_column: Column in data containing the date to match
        
    Returns:
        DataFrame with added macroeconomic features
    """
    logger.info("Adding macroeconomic features")
    result = data.copy()
    
    # Ensure dates are datetime type
    if pd.api.types.is_string_dtype(result[date_column]):
        result[date_column] = pd.to_datetime(result[date_column])
    
    if not pd.api.types.is_datetime64_dtype(macro_data['date']):
        macro_data['date'] = pd.to_datetime(macro_data['date'])
    
    # Set date as index for faster lookups
    macro_data_indexed = macro_data.set_index('date')
    
    # Function to find closest date
    def get_closest_date(date, dates):
        return dates[np.abs(dates - date).argmin()]
    
    # Get available dates
    available_dates = macro_data_indexed.index.values
    
    # Create columns for each macro indicator
    macro_indicators = macro_data_indexed.columns
    
    # Initialize new columns
    for indicator in macro_indicators:
        result[f'macro_{indicator}'] = np.nan
    
    # For each row, find the closest date in macro_data
    for idx, row in result.iterrows():
        date = row[date_column]
        closest_date = get_closest_date(date, available_dates)
        
        # Add macro indicators
        for indicator in macro_indicators:
            result.at[idx, f'macro_{indicator}'] = macro_data_indexed.loc[closest_date, indicator]
    
    logger.info(f"Added {len(macro_indicators)} macroeconomic features")
    
    return result