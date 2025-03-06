"""
Data preprocessing module for credit risk assessment.
Functions for cleaning, transforming, and preparing data.
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Dict, List, Tuple, Any, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Update src/data/preprocessing.py
def handle_missing_values(data: pd.DataFrame, 
                          categorical_cols: List[str], 
                          numerical_cols: List[str],
                          strategy: str = 'knn') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        data: DataFrame to process
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        strategy: Imputation strategy ('knn', 'mean', 'median')
        
    Returns:
        DataFrame with missing values imputed
    """
    result = data.copy()
    
    logger.info(f"Handling missing values using {strategy} strategy")
    
    # Filter column lists to only include columns that exist in the data
    existing_cat_cols = [col for col in categorical_cols if col in result.columns]
    existing_num_cols = [col for col in numerical_cols if col in result.columns]
    
    # Log any missing columns
    missing_cat_cols = [col for col in categorical_cols if col not in result.columns]
    missing_num_cols = [col for col in numerical_cols if col not in result.columns]
    
    if missing_cat_cols:
        logger.warning(f"These categorical columns from config do not exist in data: {missing_cat_cols}")
    
    if missing_num_cols:
        logger.warning(f"These numerical columns from config do not exist in data: {missing_num_cols}")
    
    # Process categorical columns
    for col in existing_cat_cols:
        if result[col].isnull().sum() > 0:
            logger.info(f"Filling missing values in {col} with 'Unknown'")
            result[col] = result[col].fillna('Unknown')
    
    # Process numerical columns
    if strategy == 'knn':
        null_cols = [col for col in existing_num_cols if result[col].isnull().sum() > 0]
        if null_cols:
            logger.info(f"Columns with null values: {null_cols}")
            # Simple imputation as a fallback
            for col in null_cols:
                result[col] = result[col].fillna(result[col].median())
    else:
        for col in existing_num_cols:
            if result[col].isnull().sum() > 0:
                logger.info(f"Filling missing values in {col} with {strategy}")
                if strategy == 'mean':
                    result[col] = result[col].fillna(result[col].mean())
                elif strategy == 'median':
                    result[col] = result[col].fillna(result[col].median())
    
    return result

def handle_outliers(data: pd.DataFrame, 
                    numerical_cols: List[str],
                    method: str = 'cap',
                    lower_percentile: float = 0.01,
                    upper_percentile: float = 0.99) -> pd.DataFrame:
    """
    Handle outliers in numerical columns
    
    Args:
        data: DataFrame to process
        numerical_cols: List of numerical columns to check for outliers
        method: Outlier handling method ('cap', 'remove')
        lower_percentile: Lower percentile cutoff
        upper_percentile: Upper percentile cutoff
        
    Returns:
        DataFrame with outliers handled
    """
    result = data.copy()
    
    logger.info(f"Handling outliers using {method} method")
    
    if method == 'cap':
        for col in numerical_cols:
            lower = result[col].quantile(lower_percentile)
            upper = result[col].quantile(upper_percentile)
            logger.info(f"Capping {col} at [{lower:.2f}, {upper:.2f}]")
            
            # Count outliers before capping
            n_lower = (result[col] < lower).sum()
            n_upper = (result[col] > upper).sum()
            if n_lower > 0 or n_upper > 0:
                logger.info(f"Found {n_lower} low outliers and {n_upper} high outliers in {col}")
            
            # Apply capping
            result[col] = result[col].clip(lower=lower, upper=upper)
    
    elif method == 'remove':
        # Mark rows with outliers
        mask = pd.Series(True, index=result.index)
        for col in numerical_cols:
            lower = result[col].quantile(lower_percentile)
            upper = result[col].quantile(upper_percentile)
            col_mask = (result[col] >= lower) & (result[col] <= upper)
            mask = mask & col_mask
            
            # Count outliers
            n_outliers = (~col_mask).sum()
            if n_outliers > 0:
                logger.info(f"Found {n_outliers} outliers in {col}")
        
        # Remove outliers
        n_before = len(result)
        result = result[mask]
        n_after = len(result)
        logger.info(f"Removed {n_before - n_after} rows containing outliers")
    
    return result

def encode_categorical_features(data: pd.DataFrame,
                              categorical_cols: List[str],
                              method: str = 'onehot') -> Tuple[pd.DataFrame, Any]:
    """
    Encode categorical features
    
    Args:
        data: DataFrame to process
        categorical_cols: List of categorical columns to encode
        method: Encoding method ('onehot', 'label', 'woe')
        
    Returns:
        Tuple of (encoded DataFrame, encoder object)
    """
    result = data.copy()
    
    logger.info(f"Encoding categorical features using {method}")
    
    # Filter to only columns that exist in the data
    existing_cat_cols = [col for col in categorical_cols if col in result.columns]
    
    if not existing_cat_cols:
        logger.warning("No categorical columns found in data")
        return result, None
    
    if method == 'onehot':
        # Use pandas get_dummies for one-hot encoding - more reliable than sklearn
        for col in existing_cat_cols:
            # Create dummies
            dummies = pd.get_dummies(result[col], prefix=col, drop_first=True)
            
            # Add to result
            result = pd.concat([result, dummies], axis=1)
        
        # Drop original categorical columns
        result = result.drop(columns=existing_cat_cols)
        
        return result, None
    
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        
        encoders = {}
        for col in existing_cat_cols:
            encoder = LabelEncoder()
            result[col] = encoder.fit_transform(result[col])
            encoders[col] = encoder
        
        return result, encoders
    
    else:
        logger.warning(f"Encoding method '{method}' not implemented, returning original data")
        return result, None
    
def scale_features(data: pd.DataFrame,
                  numerical_cols: List[str],
                  method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
    """
    Scale numerical features
    
    Args:
        data: DataFrame to process
        numerical_cols: List of numerical columns to scale
        method: Scaling method ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    result = data.copy()
    
    logger.info(f"Scaling numerical features using {method}")
    
    # Filter to only columns that exist in the data
    existing_num_cols = [col for col in numerical_cols if col in result.columns]
    
    if not existing_num_cols:
        logger.warning("No numerical columns found in data")
        return result, None
    
    # Simple min-max scaling (0-1 normalization)
    for col in existing_num_cols:
        min_val = result[col].min()
        max_val = result[col].max()
        
        # Avoid division by zero
        if max_val > min_val:
            result[col] = (result[col] - min_val) / (max_val - min_val)
        else:
            logger.warning(f"Column {col} has min==max, skipping scaling")
    
    return result, None

def prepare_training_data(data: pd.DataFrame, 
                         config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare data for model training
    
    Args:
        data: Raw DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Preparing training data")
    
    # Make a copy of the data
    df = data.copy()
    
    # Extract target variable
    target_col = config['data']['target']
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Split into training and testing sets
    test_size = config['data'].get('test_size', 0.3)
    random_state = config['data'].get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets")
    logger.info(f"Training set default rate: {y_train.mean():.2%}")
    logger.info(f"Testing set default rate: {y_test.mean():.2%}")
    
    return X_train, y_train, X_test, y_test

def handle_class_imbalance(X_train: pd.DataFrame, y_train: pd.Series,
                          method: str = 'smote',
                          sampling_strategy: Union[float, str] = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance in the training data
    
    Args:
        X_train: Training features
        y_train: Training target
        method: Resampling method ('smote', 'adasyn', 'random_over', 'random_under', 'smoteenn')
        sampling_strategy: Sampling strategy
        
    Returns:
        Tuple of (resampled X_train, resampled y_train)
    """
    logger.info(f"Handling class imbalance using {method}, original default rate: {y_train.mean():.2%}")
    
    # Import appropriate resampling technique
    if method == 'smote':
        from imblearn.over_sampling import SMOTE
        resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'adasyn':
        from imblearn.over_sampling import ADASYN
        resampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'random_over':
        from imblearn.over_sampling import RandomOverSampler
        resampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'random_under':
        from imblearn.under_sampling import RandomUnderSampler
        resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'smoteenn':
        from imblearn.combine import SMOTEENN
        resampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
    else:
        logger.error(f"Unknown resampling method: {method}")
        raise ValueError(f"Unknown resampling method: {method}")
    
    # Apply resampling
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    
    logger.info(f"After resampling: {len(X_resampled)} samples, default rate: {y_resampled.mean():.2%}")
    
    return X_resampled, y_resampled