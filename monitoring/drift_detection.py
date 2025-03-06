"""
Drift detection module for credit risk assessment.
Functions for detecting model and data drift over time.
"""
import pandas as pd
import numpy as np
import os
import logging
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriftDetector:
    """Class for detecting drift in model inputs and outputs"""
    
    def __init__(self, 
                reference_data: pd.DataFrame, 
                model_path: str,
                config_path: str,
                output_dir: str):
        """
        Initialize drift detector
        
        Args:
            reference_data: Reference data (training data)
            model_path: Path to trained model
            config_path: Path to configuration file
            output_dir: Directory to save drift reports
        """
        self.reference_data = reference_data
        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = output_dir
        
        # Load model
        self.model = joblib.load(model_path)
        
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get feature definitions
        self.numerical_cols = self.config['features']['numerical']
        self.categorical_cols = self.config['features']['categorical']
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(reference_data)
        
        # Generate reference predictions
        if 'default_flag' in reference_data.columns:
            X_ref = reference_data.drop('default_flag', axis=1)
        else:
            X_ref = reference_data
            
        self.reference_predictions = self.model.predict_proba(X_ref)[:, 1]
        
        # Calculate prediction distribution statistics
        self.prediction_stats = self._calculate_prediction_stats(self.reference_predictions)
        
        logger.info("Drift detector initialized with reference data")

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for numerical and categorical features
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Numerical features
        for col in self.numerical_cols:
            if col in data.columns:
                stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'p10': data[col].quantile(0.1),
                    'p90': data[col].quantile(0.9),
                    'missing_rate': data[col].isnull().mean()
                }
        
        # Categorical features
        for col in self.categorical_cols:
            if col in data.columns:
                value_counts = data[col].value_counts(normalize=True).to_dict()
                stats[col] = {
                    'value_counts': value_counts,
                    'n_categories': len(value_counts),
                    'most_common': data[col].value_counts().index[0] if len(value_counts) > 0 else None,
                    'missing_rate': data[col].isnull().mean()
                }
        
        return stats

    def _calculate_prediction_stats(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics for model predictions
        
        Args:
            predictions: Array of predicted probabilities
            
        Returns:
            Dictionary with prediction statistics
        """
        stats = {
            'mean': predictions.mean(),
            'std': predictions.std(),
            'min': predictions.min(),
            'max': predictions.max(),
            'median': np.median(predictions),
            'p10': np.percentile(predictions, 10),
            'p90': np.percentile(predictions, 90),
            # Calculate percentages in risk tiers
            'very_low_risk': (predictions < 0.05).mean(),
            'low_risk': ((predictions >= 0.05) & (predictions < 0.1)).mean(),
            'moderate_risk': ((predictions >= 0.1) & (predictions < 0.2)).mean(),
            'high_risk': ((predictions >= 0.2) & (predictions < 0.3)).mean(),
            'very_high_risk': (predictions >= 0.3).mean()
        }
        
        return stats

    def calculate_psi(self, expected_probs: np.ndarray, 
                     actual_probs: np.ndarray, 
                     buckets: int = 10) -> float:
        """
        Calculate Population Stability Index
        
        Args:
            expected_probs: Expected distribution (reference)
            actual_probs: Actual distribution (current)
            buckets: Number of buckets
            
        Returns:
            PSI value
        """
        # Define buckets based on expected distribution
        breaks = np.percentile(expected_probs, np.linspace(0, 100, buckets + 1))
        
        # Ensure unique breaks
        breaks = np.unique(breaks)
        
        # Count observations in each bucket
        expected_counts = np.histogram(expected_probs, bins=breaks)[0]
        actual_counts = np.histogram(actual_probs, bins=breaks)[0]
        
        # Calculate percentages
        expected_pcts = expected_counts / expected_counts.sum()
        actual_pcts = actual_counts / actual_counts.sum()
        
        # Replace zeros with small number to avoid division by zero
        expected_pcts = np.where(expected_pcts == 0, 0.0001, expected_pcts)
        actual_pcts = np.where(actual_pcts == 0, 0.0001, actual_pcts)
        
        # Calculate PSI
        psi_values = (actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)
        psi = np.sum(psi_values)
        
        return psi

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference data and current data
        
        Args:
            current_data: Current data to compare
            
        Returns:
            Dictionary with drift metrics
        """
        logger.info(f"Detecting drift on {len(current_data)} samples")
        
        # Calculate statistics for current data
        current_stats = self._calculate_statistics(current_data)
        
        # Generate predictions for current data
        if 'default_flag' in current_data.columns:
            X_current = current_data.drop('default_flag', axis=1)
            y_current = current_data['default_flag']
            has_targets = True
        else:
            X_current = current_data
            has_targets = False
            
        current_predictions = self.model.predict_proba(X_current)[:, 1]
        
        # Calculate prediction statistics
        current_prediction_stats = self._calculate_prediction_stats(current_predictions)
        
        # Calculate PSI for predictions
        prediction_psi = self.calculate_psi(self.reference_predictions, current_predictions)
        
        # Calculate drift metrics for each feature
        feature_drift = {}
        
        for col in self.numerical_cols:
            if col in current_data.columns and col in self.reference_stats:
                ref_stats = self.reference_stats[col]
                curr_stats = current_stats[col]
                
                # Calculate standardized difference in means
                mean_diff = abs(curr_stats['mean'] - ref_stats['mean'])
                pooled_std = np.sqrt((ref_stats['std']**2 + curr_stats['std']**2) / 2)
                
                if pooled_std > 0:
                    std_diff = mean_diff / pooled_std
                else:
                    std_diff = 0
                
                # Calculate KS statistic for distributions
                from scipy import stats
                
                # Get non-null values
                ref_values = self.reference_data[col].dropna().values
                curr_values = current_data[col].dropna().values
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
                else:
                    ks_stat, ks_pvalue = 0, 1
                
                feature_drift[col] = {
                    'mean_ref': ref_stats['mean'],
                    'mean_current': curr_stats['mean'],
                    'std_diff': std_diff,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'drift_detected': ks_pvalue < 0.05 or std_diff > 0.5
                }
        
        for col in self.categorical_cols:
            if col in current_data.columns and col in self.reference_stats:
                ref_stats = self.reference_stats[col]
                curr_stats = current_stats[col]
                
                # Calculate chi-square test
                from scipy import stats
                
                # Get value counts
                ref_counts = pd.Series(ref_stats['value_counts'])
                curr_counts = pd.Series(curr_stats['value_counts'])
                
                # Align categories
                all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
                ref_counts = ref_counts.reindex(all_categories, fill_value=0.0001)
                curr_counts = curr_counts.reindex(all_categories, fill_value=0.0001)
                
                # Convert to counts
                ref_n = len(self.reference_data)
                curr_n = len(current_data)
                
                ref_counts_abs = (ref_counts * ref_n).round().astype(int)
                curr_counts_abs = (curr_counts * curr_n).round().astype(int)
                
                # Ensure minimum expected counts for chi-square
                min_expected = 5
                
                valid_chi2 = all(count >= min_expected for count in np.array([ref_counts_abs, curr_counts_abs]).flatten())
                
                if valid_chi2:
                    chi2_stat, chi2_pvalue = stats.chisquare(
                        curr_counts_abs, 
                        f_exp=ref_counts_abs
                    )
                else:
                    chi2_stat, chi2_pvalue = 0, 1
                
                # Calculate Jensen-Shannon divergence
                from scipy.spatial.distance import jensenshannon
                
                js_divergence = jensenshannon(ref_counts, curr_counts)
                
                feature_drift[col] = {
                    'chi2_statistic': float(chi2_stat),
                    'chi2_pvalue': float(chi2_pvalue),
                    'js_divergence': float(js_divergence),
                    'drift_detected': chi2_pvalue < 0.05 or js_divergence > 0.2
                }
        
        # Calculate model performance metrics if targets are available
        performance_metrics = {}
        
        if has_targets:
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
            
            try:
                # Classification threshold
                threshold = 0.5
                
                # Calculate metrics
                auc = roc_auc_score(y_current, current_predictions)
                precision = precision_score(y_current, current_predictions >= threshold)
                recall = recall_score(y_current, current_predictions >= threshold)
                f1 = f1_score(y_current, current_predictions >= threshold)
                
                performance_metrics = {
                    'auc': auc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            except Exception as e:
                logger.error(f"Error calculating performance metrics: {str(e)}")
                performance_metrics = {}
        
        # Compile drift report
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'n_reference': len(self.reference_data),
            'n_current': len(current_data),
            'prediction_psi': prediction_psi,
            'prediction_drift_detected': prediction_psi > 0.25,
            'feature_drift': feature_drift,
            'any_feature_drift_detected': any(v.get('drift_detected', False) for v in feature_drift.values()),
            'reference_prediction_stats': self.prediction_stats,
            'current_prediction_stats': current_prediction_stats,
            'performance_metrics': performance_metrics
        }
        
        # Save drift report
        self._save_drift_report(drift_report)
        
        # Log summary
        logger.info(f"Drift detection complete. PSI: {prediction_psi:.4f}, "
                   f"Features with drift: {sum(1 for v in feature_drift.values() if v.get('drift_detected', False))}")
        
        return drift_report

    def _save_drift_report(self, drift_report: Dict[str, Any]) -> None:
        """
        Save drift report to file
        
        Args:
            drift_report: Drift report to save
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"drift_report_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save report
        with open(filepath, 'w') as f:
            json.dump(drift_report, f, indent=2)
        
        logger.info(f"Drift report saved to {filepath}")

    def trigger_retraining(self, drift_report: Dict[str, Any], 
                         retraining_threshold: float = 0.25) -> bool:
        """
        Determine if retraining is needed based on drift report
        
        Args:
            drift_report: Drift report
            retraining_threshold: PSI threshold for retraining
            
        Returns:
            Boolean indicating whether retraining is needed
        """
        # Check if PSI exceeds threshold
        prediction_psi = drift_report.get('prediction_psi', 0)
        
        # Check if significant feature drift is detected
        feature_drift = drift_report.get('feature_drift', {})
        significant_feature_drift = any(
            v.get('drift_detected', False) for v in feature_drift.values()
        )
        
        # Check if performance has degraded
        performance_metrics = drift_report.get('performance_metrics', {})
        performance_degraded = False
        
        if 'auc' in performance_metrics:
            # Compare with a threshold (could be more sophisticated in practice)
            performance_degraded = performance_metrics['auc'] < 0.7
        
        # Determine if retraining is needed
        retraining_needed = (
            prediction_psi > retraining_threshold or
            (significant_feature_drift and performance_degraded)
        )
        
        if retraining_needed:
            logger.info(f"Retraining recommended: PSI={prediction_psi:.4f}, "
                       f"Feature drift={significant_feature_drift}, "
                       f"Performance degraded={performance_degraded}")
        else:
            logger.info(f"No retraining needed: PSI={prediction_psi:.4f}, "
                       f"Feature drift={significant_feature_drift}, "
                       f"Performance degraded={performance_degraded}")
        
        return retraining_needed