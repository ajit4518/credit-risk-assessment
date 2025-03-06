"""
Data acquisition module for credit risk assessment.
Functions to load data from various sources.
"""
import os
import pandas as pd
import yaml
import logging
from typing import Tuple, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw training and testing data
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (training_data, testing_data)
    """
    if config is None:
        config = load_config()
    
    paths = config['paths']
    data_config = config['data']
    
    train_path = os.path.join(paths['raw_data'], data_config['train_file'])
    test_path = os.path.join(paths['raw_data'], data_config['test_file'])
    
    logger.info(f"Loading training data from {train_path}")
    try:
        train_data = pd.read_csv(train_path)
        logger.info(f"Loaded training data with shape {train_data.shape}")
    except FileNotFoundError:
        logger.error(f"Training file not found at {train_path}")
        logger.info("Creating dummy data for development purposes")
        train_data = create_dummy_data(n_samples=10000, random_state=config['data']['random_state'])
    
    try:
        test_data = pd.read_csv(test_path)
        logger.info(f"Loaded test data with shape {test_data.shape}")
    except FileNotFoundError:
        logger.error(f"Test file not found at {test_path}")
        logger.info("Creating dummy test data")
        test_data = create_dummy_data(n_samples=3000, random_state=config['data']['random_state']+1)
    
    return train_data, test_data

def load_external_credit_data(customer_ids: list) -> pd.DataFrame:
    """
    Simulate loading external credit bureau data
    
    Args:
        customer_ids: List of customer IDs to retrieve data for
        
    Returns:
        DataFrame with credit bureau data
    """
    # In a real implementation, this might call an API or load from a database
    logger.info(f"Loading credit bureau data for {len(customer_ids)} customers")
    
    # Create dummy credit bureau data
    import numpy as np
    np.random.seed(42)
    
    credit_data = pd.DataFrame({
        'customer_id': customer_ids,
        'credit_score': np.random.normal(700, 100, size=len(customer_ids)).clip(300, 850).astype(int),
        'num_inquiries_12m': np.random.poisson(2, size=len(customer_ids)),
        'num_delinquencies_24m': np.random.poisson(0.5, size=len(customer_ids)),
        'oldest_credit_line_age': np.random.gamma(shape=5, scale=24, size=len(customer_ids)).astype(int),
        'total_credit_lines': np.random.poisson(5, size=len(customer_ids))
    })
    
    return credit_data

def load_macroeconomic_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load macroeconomic data for a specific time period
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with macroeconomic indicators
    """
    logger.info(f"Loading macroeconomic data from {start_date} to {end_date}")
    
    # In a real implementation, this might call FRED API or similar
    # For now, create simulated data
    
    import numpy as np
    from pandas.tseries.offsets import MonthEnd
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    date_range = pd.date_range(start=start, end=end, freq='M')
    
    np.random.seed(42)
    macro_data = pd.DataFrame({
        'date': date_range,
        'gdp_growth': np.random.normal(0.02, 0.01, size=len(date_range)) / 12,  # Monthly growth rate
        'unemployment_rate': np.random.normal(0.05, 0.01, size=len(date_range)),
        'fed_funds_rate': np.random.normal(0.02, 0.005, size=len(date_range)),
        'hpi_yoy_change': np.random.normal(0.03, 0.02, size=len(date_range))
    })
    
    return macro_data

def create_dummy_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Create synthetic credit data for development purposes
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic credit data
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(random_state)
    
    # Generate customer IDs
    customer_ids = [f'CUST{i:06d}' for i in range(1, n_samples + 1)]
    
    # Generate loan IDs
    loan_ids = [f'LOAN{i:08d}' for i in range(1, n_samples + 1)]
    
    # Generate origination dates (last 3 years)
    today = datetime.now()
    start_date = today - timedelta(days=3*365)
    days_range = (today - start_date).days
    origination_dates = [start_date + timedelta(days=np.random.randint(0, days_range)) for _ in range(n_samples)]
    origination_dates_str = [d.strftime('%Y-%m-%d') for d in origination_dates]
    
    # Generate features that influence default
    income = np.random.lognormal(mean=11, sigma=0.5, size=n_samples)  # Income around $60k
    debt_to_income = np.random.beta(2, 5, size=n_samples) * 0.5  # DTI mostly below 0.5
    loan_amount = np.random.lognormal(mean=10, sigma=1, size=n_samples)  # Loan amounts ~$20k-30k
    interest_rate = np.random.normal(0.05, 0.02, size=n_samples)  # Interest rates around 5%
    credit_score = np.random.normal(700, 100, size=n_samples).clip(300, 850)  # Credit scores
    
    # Categorical features
    employment_status = np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired'], 
                                         size=n_samples, p=[0.7, 0.15, 0.05, 0.1])
    housing_status = np.random.choice(['Own', 'Mortgage', 'Rent', 'Other'], 
                                      size=n_samples, p=[0.3, 0.4, 0.25, 0.05])
    product_type = np.random.choice(['Personal Loan', 'Auto Loan', 'Credit Card', 'Mortgage'], 
                                    size=n_samples, p=[0.3, 0.3, 0.2, 0.2])
    purpose = np.random.choice(['Debt Consolidation', 'Home Improvement', 'Major Purchase', 
                                'Business', 'Medical', 'Other'], 
                               size=n_samples, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
    
    # Create logit for default
    logit = (-0.5 + 
             -0.7 * np.log(income) / 11 + 
             5 * debt_to_income + 
             0.3 * np.log(loan_amount) / 10 +
             10 * interest_rate +
             -0.01 * (credit_score - 700) / 100 +
             0.5 * (employment_status == 'Unemployed') +
             0.3 * (housing_status == 'Rent') +
             -0.2 * (product_type == 'Mortgage') +
             0.1 * (purpose == 'Debt Consolidation') +
             np.random.normal(0, 1, size=n_samples))
    
    # Convert to probability and generate default flag
    default_prob = 1 / (1 + np.exp(-logit))
    default_flag = (np.random.random(size=n_samples) < default_prob).astype(int)
    
    # Create dataframe
    data = pd.DataFrame({
        'loan_id': loan_ids,
        'customer_id': customer_ids,
        'origination_date': origination_dates_str,
        'income': income,
        'debt_to_income_ratio': debt_to_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'credit_score': credit_score,
        'employment_status': employment_status,
        'housing_status': housing_status,
        'product_type': product_type,
        'purpose': purpose,
        'default_flag': default_flag,
        'default_probability': default_prob
    })
    
    # Add payment history
    payment_history = []
    for i in range(n_samples):
        num_payments = np.random.randint(1, 24)
        avg_payment_ratio = 0.9 + 0.2 * np.random.random() if default_flag[i] == 0 else 0.7 + 0.3 * np.random.random()
        max_days_late = np.random.randint(0, 10) if default_flag[i] == 0 else np.random.randint(10, 90)
        pct_late_payments = np.random.beta(1, 10) if default_flag[i] == 0 else np.random.beta(5, 10)
        
        payment_history.append({
            'loan_id': loan_ids[i],
            'num_payments': num_payments,
            'avg_payment_ratio': avg_payment_ratio,
            'max_days_late': max_days_late,
            'pct_late_payments': pct_late_payments
        })
    
    payment_history_df = pd.DataFrame(payment_history)
    data = data.merge(payment_history_df, on='loan_id')
    
    logger.info(f"Generated synthetic credit dataset with {n_samples} samples")
    logger.info(f"Default rate: {data['default_flag'].mean():.2%}")
    
    return data