"""
Exploratory data analysis module for credit risk assessment.
Functions for generating visualizations and statistical summaries.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Any, Union, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_target_distribution(data: pd.DataFrame, target_col: str = 'default_flag',
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of the target variable
    
    Args:
        data: DataFrame containing the data
        target_col: Name of the target column
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting target distribution")
    
    # Count values
    target_counts = data[target_col].value_counts().sort_index()
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(target_counts.index.astype(str), target_counts.values, color=['#3498db', '#e74c3c'])
    
    # Add percentages
    total = len(data)
    for i, count in enumerate(target_counts):
        percentage = count / total * 100
        ax.text(i, count + 0.1, f'{percentage:.1f}%', ha='center')
    
    # Add labels and title
    ax.set_xlabel(target_col)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {target_col}')
    
    # Add total count
    ax.text(0.5, 0.9, f'Total samples: {total}', transform=ax.transAxes, ha='center')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_numerical_distributions(data: pd.DataFrame, 
                              numerical_cols: List[str],
                              target_col: str = 'default_flag',
                              bins: int = 30,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distributions of numerical features by target
    
    Args:
        data: DataFrame containing the data
        numerical_cols: List of numerical columns to plot
        target_col: Name of the target column
        bins: Number of bins for histograms
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting distributions for {len(numerical_cols)} numerical features")
    
    # Calculate number of rows and columns for subplots
    n_cols = 2
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    # Flatten axes array for easy iteration
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes[0], axes[1]]
    
    # Plot each feature
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            ax = axes[i]
            
            # Create separate histograms by target
            for target_val in sorted(data[target_col].unique()):
                subset = data[data[target_col] == target_val]
                ax.hist(subset[col], bins=bins, alpha=0.5, 
                       label=f'{target_col}={target_val}', density=True)
            
            # Add labels and legend
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {col} by {target_col}')
            ax.legend()
    
    # Hide unused subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_categorical_distributions(data: pd.DataFrame, 
                                categorical_cols: List[str],
                                target_col: str = 'default_flag',
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distributions of categorical features by target
    
    Args:
        data: DataFrame containing the data
        categorical_cols: List of categorical columns to plot
        target_col: Name of the target column
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting distributions for {len(categorical_cols)} categorical features")
    
    # Calculate number of rows and columns for subplots
    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    # Flatten axes array for easy iteration
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes[0], axes[1]]
    
    # Plot each feature
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            ax = axes[i]
            
            # Create crosstab
            ct = pd.crosstab(data[col], data[target_col])
            
            # Convert to percentages
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            
            # Plot stacked bar chart
            ct_pct.plot(kind='bar', stacked=True, ax=ax, 
                       colormap='coolwarm')
            
            # Add count annotations
            for j, p in enumerate(ax.patches):
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy() 
                
                # Only add text if the segment is large enough
                if height > 5:
                    ax.text(x + width/2, y + height/2, f'{height:.1f}%', 
                           ha='center', va='center')
            
            # Add labels and title
            ax.set_xlabel(col)
            ax.set_ylabel('Percentage')
            ax.set_title(f'Distribution of {target_col} by {col}')
            ax.legend(title=target_col)
    
    # Hide unused subplots
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_correlation_heatmap(data: pd.DataFrame, 
                          numerical_cols: List[str],
                          method: str = 'pearson',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation heatmap for numerical features
    
    Args:
        data: DataFrame containing the data
        numerical_cols: List of numerical columns to include
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting correlation heatmap using {method} method")
    
    # Calculate correlation matrix
    corr = data[numerical_cols].corr(method=method)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Generate mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
               annot=True, fmt='.2f', square=True, linewidths=.5, ax=ax)
    
    # Add title
    ax.set_title(f'Feature Correlation Matrix ({method.capitalize()})')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_feature_importances(feature_names: List[str], 
                          importances: np.ndarray,
                          title: str = 'Feature Importances',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importances
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting feature importances")
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot top 20 features (or all if less than 20)
    n_features = min(20, len(importance_df))
    sns.barplot(x='importance', y='feature', data=importance_df.head(n_features), ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_default_rates_by_feature(data: pd.DataFrame, 
                               feature_col: str,
                               target_col: str = 'default_flag',
                               bins: int = 10,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot default rates by feature bins
    
    Args:
        data: DataFrame containing the data
        feature_col: Feature column to analyze
        target_col: Target column
        bins: Number of bins for numerical features
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting default rates by {feature_col}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Handle numerical vs categorical features differently
    if pd.api.types.is_numeric_dtype(data[feature_col]):
        # Create bins for numerical feature
        data_binned = data.copy()
        data_binned[f'{feature_col}_bin'] = pd.cut(data[feature_col], bins=bins)
        
        # Calculate default rate by bin
        default_rates = data_binned.groupby(f'{feature_col}_bin')[target_col].mean() * 100
        counts = data_binned.groupby(f'{feature_col}_bin').size()
        
        # Plot default rates
        default_rates.plot(kind='bar', ax=ax, color='#3498db')
        
        # Add count labels
        for i, (rate, count) in enumerate(zip(default_rates, counts)):
            ax.text(i, rate + 0.5, f'n={count}', ha='center')
        
        ax.set_xlabel(feature_col)
    else:
        # Calculate default rate by category
        default_rates = data.groupby(feature_col)[target_col].mean() * 100
        counts = data.groupby(feature_col).size()
        
        # Sort by default rate
        default_rates = default_rates.sort_values(ascending=False)
        counts = counts.reindex(default_rates.index)
        
        # Plot default rates
        default_rates.plot(kind='bar', ax=ax, color='#3498db')
        
        # Add count labels
        for i, (rate, count) in enumerate(zip(default_rates, counts)):
            ax.text(i, rate + 0.5, f'n={count}', ha='center')
        
        ax.set_xlabel(feature_col)
    
    # Add labels and title
    ax.set_ylabel(f'{target_col} Rate (%)')
    ax.set_title(f'{target_col.capitalize()} Rate by {feature_col}')
    
    # Add average line
    avg_rate = data[target_col].mean() * 100
    ax.axhline(y=avg_rate, color='r', linestyle='--', 
              label=f'Average: {avg_rate:.2f}%')
    
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_distribution_by_time(data: pd.DataFrame, 
                           feature_col: str,
                           time_col: str,
                           target_col: str = 'default_flag',
                           freq: str = 'M',
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature or target distribution over time
    
    Args:
        data: DataFrame containing the data
        feature_col: Feature column to plot
        time_col: Time/date column
        target_col: Target column
        freq: Time frequency for resampling ('D', 'W', 'M', 'Q', 'Y')
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting {feature_col} distribution over time")
    
    # Ensure time column is datetime
    data = data.copy()
    if not pd.api.types.is_datetime64_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col])
    
    # Set time column as index
    data = data.set_index(time_col)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # If feature_col is the target, plot default rate over time
    if feature_col == target_col:
        # Resample and calculate default rate
        default_rate = data[target_col].resample(freq).mean() * 100
        volume = data[target_col].resample(freq).count()
        
        # Plot default rate
        default_rate.plot(ax=ax1, color='#e74c3c', marker='o')
        ax1.set_ylabel(f'{target_col.capitalize()} Rate (%)', color='#e74c3c')
        ax1.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Add volume on secondary axis
        ax2 = ax1.twinx()
        volume.plot(ax=ax2, color='#3498db', alpha=0.5, kind='bar')
        ax2.set_ylabel('Volume', color='#3498db')
        ax2.tick_params(axis='y', labelcolor='#3498db')
        
        # Set title
        ax1.set_title(f'{target_col.capitalize()} Rate Over Time')
    else:
        # If numerical feature, plot average over time
        if pd.api.types.is_numeric_dtype(data[feature_col]):
            # Resample and calculate mean
            feature_mean = data[feature_col].resample(freq).mean()
            feature_std = data[feature_col].resample(freq).std()
            volume = data[feature_col].resample(freq).count()
            
            # Plot mean
            feature_mean.plot(ax=ax1, color='#e74c3c', marker='o')
            
            # Add confidence interval
            ax1.fill_between(
                feature_mean.index,
                feature_mean - feature_std,
                feature_mean + feature_std,
                color='#e74c3c', alpha=0.2
            )
            
            ax1.set_ylabel(feature_col, color='#e74c3c')
            ax1.tick_params(axis='y', labelcolor='#e74c3c')
            
            # Add volume on secondary axis
            ax2 = ax1.twinx()
            volume.plot(ax=ax2, color='#3498db', alpha=0.5, kind='bar')
            ax2.set_ylabel('Volume', color='#3498db')
            ax2.tick_params(axis='y', labelcolor='#3498db')
            
            # Set title
            ax1.set_title(f'{feature_col} Over Time')
        else:
            # For categorical features, plot stacked bar chart
            # Create pivot table for stacked bar
            pivot = pd.crosstab(
                index=data.index.to_period(freq),
                columns=data[feature_col],
                normalize='index'
            ) * 100
            
            # Convert period index back to timestamp for plotting
            pivot.index = pivot.index.to_timestamp()
            
            # Plot stacked bar
            pivot.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
            
            ax1.set_ylabel('Percentage (%)')
            ax1.set_title(f'Distribution of {feature_col} Over Time')
            ax1.legend(title=feature_col)
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_missing_values(data: pd.DataFrame,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot missing values heatmap
    
    Args:
        data: DataFrame to analyze
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting missing values heatmap")
    
    # Calculate missing values
    missing = data.isnull().sum() / len(data) * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    
    # If no missing values, create empty plot with message
    if len(missing) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No missing values", fontsize=14, ha='center')
        ax.set_title("Missing Values")
        ax.set_axis_off()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot missing values
    sns.barplot(x=missing.index, y=missing.values, ax=ax)
    
    # Add labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.1,
               f'{height:.1f}%', ha="center")
    
    # Add labels and title
    ax.set_xlabel('Features')
    ax.set_ylabel('Missing Values (%)')
    ax.set_title('Missing Values by Feature')
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def create_summary_statistics(data: pd.DataFrame, 
                           numerical_cols: List[str],
                           categorical_cols: List[str],
                           target_col: str = 'default_flag') -> Dict[str, pd.DataFrame]:
    """
    Create summary statistics tables
    
    Args:
        data: DataFrame to analyze
        numerical_cols: List of numerical columns
        categorical_cols: List of categorical columns
        target_col: Target column name
        
    Returns:
        Dictionary with summary DataFrames
    """
    logger.info("Creating summary statistics")
    
    # Create summaries dictionary
    summaries = {}
    
    # Overall summary
    logger.info("Calculating overall summary")
    overall = pd.DataFrame({
        'total_records': len(data),
        'default_rate': data[target_col].mean() * 100,
        'missing_rate': data.isnull().mean().mean() * 100,
        'num_features': len(data.columns),
        'num_numerical': len(numerical_cols),
        'num_categorical': len(categorical_cols)
    }, index=[0])
    
    summaries['overall'] = overall
    
    # Numerical features summary
    logger.info("Calculating numerical features summary")
    num_summary = data[numerical_cols].describe().T
    num_summary['missing_rate'] = data[numerical_cols].isnull().mean() * 100
    
    # Add correlation with target
    target_corr = {}
    for col in numerical_cols:
        target_corr[col] = data[[col, target_col]].corr().iloc[0, 1]
    
    num_summary['target_correlation'] = pd.Series(target_corr)
    
    # Calculate information value if target is binary
    if data[target_col].nunique() == 2:
        import scipy.stats as stats
        
        iv_values = {}
        for col in numerical_cols:
            try:
                # Create 10 bins (or fewer for low-cardinality features)
                n_bins = min(10, data[col].nunique())
                data['temp_bin'] = pd.qcut(data[col], n_bins, duplicates='drop')
                
                # Calculate IV
                crosstab = pd.crosstab(data['temp_bin'], data[target_col])
                
                # Get counts for good and bad
                good = crosstab[0]
                bad = crosstab[1]
                
                # Calculate WoE and IV
                good_dist = good / good.sum()
                bad_dist = bad / bad.sum()
                
                # Replace zeros to avoid division by zero
                good_dist = good_dist.replace(0, 0.001)
                bad_dist = bad_dist.replace(0, 0.001)
                
                woe = np.log(good_dist / bad_dist)
                iv = ((good_dist - bad_dist) * woe).sum()
                
                iv_values[col] = iv
            except:
                iv_values[col] = np.nan
            
            # Drop temporary bin column
            if 'temp_bin' in data.columns:
                data = data.drop('temp_bin', axis=1)
        
        num_summary['information_value'] = pd.Series(iv_values)
    
    summaries['numerical'] = num_summary
    
    # Categorical features summary
    logger.info("Calculating categorical features summary")
    cat_summary = pd.DataFrame(index=categorical_cols)
    
    # Add basic stats
    cat_summary['unique_values'] = [data[col].nunique() for col in categorical_cols]
    cat_summary['missing_rate'] = data[categorical_cols].isnull().mean() * 100
    cat_summary['most_common'] = [data[col].value_counts().index[0] if data[col].nunique() > 0 else None for col in categorical_cols]
    cat_summary['most_common_pct'] = [data[col].value_counts(normalize=True).iloc[0] * 100 if data[col].nunique() > 0 else None for col in categorical_cols]
    
    # Add chi-square stats for relationship with target
    chi2_values = {}
    p_values = {}
    
    for col in categorical_cols:
        if data[col].nunique() > 0:
            # Create contingency table
            contingency = pd.crosstab(data[col], data[target_col])
            
            # Calculate chi-square
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            
            chi2_values[col] = chi2
            p_values[col] = p
        else:
            chi2_values[col] = np.nan
            p_values[col] = np.nan
    
    cat_summary['chi2_statistic'] = pd.Series(chi2_values)
    cat_summary['p_value'] = pd.Series(p_values)
    
    # Add Cramer's V for effect size
    cramer_v = {}
    for col in categorical_cols:
        if chi2_values[col] and not np.isnan(chi2_values[col]):
            # Create contingency table
            contingency = pd.crosstab(data[col], data[target_col])
            
            # Calculate Cramer's V
            n = contingency.sum().sum()
            phi2 = chi2_values[col] / n
            r, k = contingency.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            cramer_v[col] = np.sqrt(phi2corr / min(kcorr-1, rcorr-1))
        else:
            cramer_v[col] = np.nan
    
    cat_summary['cramers_v'] = pd.Series(cramer_v)
    
    summaries['categorical'] = cat_summary
    
    return summaries