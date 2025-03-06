#!/usr/bin/env python3
"""
Main entry point for the Credit Risk Assessment project.
This script allows you to run various components of the system through command-line arguments.
"""
import os
import sys
import argparse
import logging
import yaml
import importlib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

# Ensure logs directory exists before setting up logging
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/credit_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("credit_risk")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories from configuration"""
    paths = config.get('paths', {})
    for path_name, path_value in paths.items():
        os.makedirs(path_value, exist_ok=True)
        logger.info(f"Created directory: {path_value}")

def process_data(config: Dict[str, Any]) -> None:
    """Process and prepare data for model training"""
    logger.info("Starting data processing")
    
    try:
        # Import data processing modules
        from src.data.acquisition import load_data
        from src.data.preprocessing import (
            handle_missing_values, 
            handle_outliers,
            encode_categorical_features,
            scale_features,
            prepare_training_data
        )
        from src.data.features import (
            create_payment_features,
            create_time_features,
            create_interaction_features,
            create_risk_flags
        )
        
        # Load raw data
        train_data, test_data = load_data(config)
        logger.info(f"Loaded training data: {train_data.shape} and test data: {test_data.shape}")
        
        # Get column definitions
        categorical_cols = config['features']['categorical']
        numerical_cols = config['features']['numerical']
        
        # Process training data
        train_data = handle_missing_values(train_data, categorical_cols, numerical_cols)
        train_data = handle_outliers(train_data, numerical_cols)
        
        # Feature engineering
        train_data = create_time_features(train_data, config['features'].get('date_columns', ['origination_date'])[0])
        train_data = create_interaction_features(train_data)
        train_data = create_risk_flags(train_data)
        
        # Encode categorical features
        train_data, encoders = encode_categorical_features(
            train_data, 
            categorical_cols, 
            method=config['preprocessing'].get('encoding_method', 'onehot')
        )
        
        # Scale numerical features
        train_data, scalers = scale_features(
            train_data,
            [col for col in train_data.columns if col in numerical_cols or col.startswith('time_') or col.startswith('interaction_')],
            method=config['preprocessing'].get('scaling_method', 'standard')
        )
        
        # Process test data similarly
        test_data = handle_missing_values(test_data, categorical_cols, numerical_cols)
        test_data = handle_outliers(test_data, numerical_cols)
        test_data = create_time_features(test_data, config['features'].get('date_columns', ['origination_date'])[0])
        test_data = create_interaction_features(test_data)
        test_data = create_risk_flags(test_data)
        
        # Save processed data
        processed_train_path = os.path.join(config['paths']['processed_data'], 'processed_train.csv')
        processed_test_path = os.path.join(config['paths']['processed_data'], 'processed_test.csv')
        
        train_data.to_csv(processed_train_path, index=False)
        test_data.to_csv(processed_test_path, index=False)
        
        logger.info(f"Saved processed training data to {processed_train_path}")
        logger.info(f"Saved processed test data to {processed_test_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}", exc_info=True)
        return False

def train_model(config: Dict[str, Any]) -> None:
    """Train the credit risk model"""
    logger.info("Starting model training")
    
    try:
        # Import model training modules
        from src.models.train import (
            load_model_config,
            train_logistic_regression,
            train_gradient_boosting,
            train_random_forest,
            train_xgboost,
            train_ensemble,
            save_model
        )
        
        # Load model configuration
        model_config = load_model_config(config.get('model_config_path', 'config/model_config.yaml'))
        logger.info("Loaded model configuration")
        
        # Load preprocessed data
        processed_train_path = os.path.join(config['paths']['processed_data'], 'processed_train.csv')
        train_data = pd.read_csv(processed_train_path)
        logger.info(f"Loaded processed training data: {train_data.shape}")
        
        # Prepare training data
        target_col = config['data']['target']
        X_train = train_data.drop(columns=[target_col])
        
        # IMPORTANT: Filter out ID and non-numeric columns
        # First, identify ID columns and other string columns
        id_columns = ['loan_id', 'customer_id']  # Add any other ID columns here
        string_columns = []
        
        for col in X_train.columns:
            # Check if column name contains 'id' or 'date'
            if 'id' in col.lower() or 'date' in col.lower():
                string_columns.append(col)
            # Check column data type
            elif X_train[col].dtype == 'object':
                string_columns.append(col)
        
        # Remove these columns from the training data
        columns_to_drop = list(set(string_columns + id_columns))
        columns_to_drop = [col for col in columns_to_drop if col in X_train.columns]
        
        if columns_to_drop:
            logger.info(f"Removing non-numeric columns for training: {columns_to_drop}")
            X_train = X_train.drop(columns=columns_to_drop)
        
        y_train = train_data[target_col]
        
        logger.info(f"Training set: {X_train.shape}, target distribution: {y_train.value_counts(normalize=True)}")
        
        # Determine which model to train based on config
        model_type = config['model'].get('type', 'ensemble')
        logger.info(f"Training model type: {model_type}")
        
        if model_type == 'logistic_regression':
            model = train_logistic_regression(X_train, y_train, model_config)
        elif model_type == 'gradient_boosting':
            model = train_gradient_boosting(X_train, y_train, model_config)
        elif model_type == 'random_forest':
            model = train_random_forest(X_train, y_train, model_config)
        elif model_type == 'xgboost':
            model = train_xgboost(X_train, y_train, model_config)
        elif model_type == 'ensemble':
            model = train_ensemble(X_train, y_train, model_config)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return False
        
        # Save trained model
        model_path = config['paths']['models']
        save_model(model, model_path, 'credit_risk_model')
        
        # Also save feature names for future reference
        import json
        with open(os.path.join(model_path, 'feature_names.json'), 'w') as f:
            json.dump(list(X_train.columns), f)
        
        logger.info("Model training complete")
        return True
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        return False

def evaluate_model(config: Dict[str, Any]) -> None:
    """Evaluate the trained model"""
    logger.info("Starting model evaluation")
    
    try:
        # Import evaluation modules
        from src.models.evaluate import (
            evaluate_classification_metrics,
            plot_roc_curve,
            plot_precision_recall_curve,
            plot_confusion_matrix,
            plot_score_distribution,
            plot_calibration_curve,
            analyze_model_fairness
        )
        import joblib
        import json
        
        # Load trained model
        model_path = os.path.join(config['paths']['models'], 'credit_risk_model.pkl')
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load feature names used during training
        feature_names_path = os.path.join(config['paths']['models'], 'feature_names.json')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                training_features = json.load(f)
            logger.info(f"Loaded {len(training_features)} feature names used during training")
        else:
            logger.warning("Feature names file not found. Evaluation may fail due to feature mismatch.")
            training_features = None
        
        # Load test data
        processed_test_path = os.path.join(config['paths']['processed_data'], 'processed_test.csv')
        test_data = pd.read_csv(processed_test_path)
        logger.info(f"Loaded test data: {test_data.shape}")
        
        # Prepare test data
        target_col = config['data']['target']
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]
        
        # Critical step: Ensure X_test has exactly the same columns as training data
        if training_features:
            # First, identify missing features
            missing_features = [f for f in training_features if f not in X_test.columns]
            if missing_features:
                logger.warning(f"Adding {len(missing_features)} features missing from test data")
                for feature in missing_features:
                    X_test[feature] = 0  # Add missing columns with zeros
            
            # Second, remove extra features
            extra_features = [f for f in X_test.columns if f not in training_features]
            if extra_features:
                logger.warning(f"Removing {len(extra_features)} extra features from test data: {extra_features[:5]}...")
                X_test = X_test.drop(columns=extra_features)
            
            # Third, ensure column order is identical
            X_test = X_test[training_features]
            logger.info(f"Aligned test features with training features, final shape: {X_test.shape}")
        
        # Create reports directory
        reports_dir = config['paths'].get('reports', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Evaluate model
        metrics = evaluate_classification_metrics(model, X_test, y_test)
        logger.info(f"Model performance metrics: {metrics}")
        
        # Create evaluation plots
        plot_roc_curve(model, X_test, y_test, save_path=os.path.join(reports_dir, 'roc_curve.png'))
        plot_precision_recall_curve(model, X_test, y_test, save_path=os.path.join(reports_dir, 'precision_recall_curve.png'))
        plot_confusion_matrix(model, X_test, y_test, save_path=os.path.join(reports_dir, 'confusion_matrix.png'))
        plot_score_distribution(model, X_test, y_test, save_path=os.path.join(reports_dir, 'score_distribution.png'))
        plot_calibration_curve(model, X_test, y_test, save_path=os.path.join(reports_dir, 'calibration_curve.png'))
        
        # Check for protected attributes in the config
        protected_cols = config.get('fairness', {}).get('protected_attributes', [])
        if protected_cols:
            # Only analyze fairness if protected attributes are available
            available_protected = [col for col in protected_cols if col in X_test.columns]
            if available_protected:
                analyze_model_fairness(model, X_test, y_test, available_protected, 
                                      save_path=os.path.join(reports_dir, 'fairness_metrics.csv'))
        
        # Save overall metrics to file
        with open(os.path.join(reports_dir, 'model_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Evaluation complete. Reports saved to {reports_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
        return False

def start_api(config: Dict[str, Any]) -> None:
    """Start the prediction API server"""
    logger.info("Starting API server")
    
    try:
        import uvicorn
        from api.app import app as api_app  # Import the FastAPI app
        
        host = config.get('api', {}).get('host', '0.0.0.0')
        port = config.get('api', {}).get('port', 8000)
        
        logger.info(f"API server starting on http://{host}:{port}")
        uvicorn.run(api_app, host=host, port=port)
        
        return True
    
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}", exc_info=True)
        return False

def start_monitoring(config: Dict[str, Any]) -> None:
    """Start the monitoring dashboard"""
    logger.info("Starting monitoring dashboard")
    
    try:
        from monitoring.dashboard import MonitoringDashboard
        
        # Get monitoring configuration
        drift_reports_dir = os.path.join(config['paths'].get('monitoring', 'monitoring'), 'drift')
        predictions_dir = os.path.join(config['paths'].get('monitoring', 'monitoring'), 'predictions')
        
        # Ensure directories exist
        os.makedirs(drift_reports_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Create and start dashboard
        dashboard = MonitoringDashboard(
            drift_reports_dir=drift_reports_dir,
            predictions_dir=predictions_dir,
            config_path=config.get('config_path', 'config/config.yaml')
        )
        
        host = config.get('monitoring', {}).get('host', '0.0.0.0')
        port = config.get('monitoring', {}).get('port', 8050)
        
        logger.info(f"Monitoring dashboard starting on http://{host}:{port}")
        dashboard.run(host=host, port=port, debug=False)
        
        return True
    
    except Exception as e:
        logger.error(f"Error starting monitoring dashboard: {str(e)}", exc_info=True)
        return False

def run_drift_detection(config: Dict[str, Any]) -> None:
    """Run drift detection on new data"""
    logger.info("Running drift detection")
    
    try:
        from monitoring.drift_detection import DriftDetector
        import joblib
        
        # Load reference data
        processed_train_path = os.path.join(config['paths']['processed_data'], 'processed_train.csv')
        reference_data = pd.read_csv(processed_train_path)
        
        # Get path to current data
        current_data_path = config.get('drift_detection', {}).get('current_data_path')
        if not current_data_path:
            logger.error("Current data path not specified in config")
            return False
        
        # Load current data
        current_data = pd.read_csv(current_data_path)
        logger.info(f"Loaded current data for drift detection: {current_data.shape}")
        
        # Initialize drift detector
        model_path = os.path.join(config['paths']['models'], 'credit_risk_model.pkl')
        drift_output_dir = os.path.join(config['paths'].get('monitoring', 'monitoring'), 'drift')
        os.makedirs(drift_output_dir, exist_ok=True)
        
        detector = DriftDetector(
            reference_data=reference_data,
            model_path=model_path,
            config_path=config.get('config_path', 'config/config.yaml'),
            output_dir=drift_output_dir
        )
        
        # Run drift detection
        drift_report = detector.detect_drift(current_data)
        
        # Check if retraining is needed
        retraining_needed = detector.trigger_retraining(
            drift_report, 
            retraining_threshold=config.get('drift_detection', {}).get('retraining_threshold', 0.25)
        )
        
        if retraining_needed:
            logger.warning("Drift detected - model retraining recommended")
        
        logger.info("Drift detection complete")
        return True
    
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}", exc_info=True)
        return False

def run_full_workflow(config: Dict[str, Any]) -> None:
    """Run the complete workflow: process data, train, evaluate, and optionally start services"""
    
    # Process data
    if not process_data(config):
        logger.error("Data processing failed, stopping workflow")
        return False
    
    # Train model
    if not train_model(config):
        logger.error("Model training failed, stopping workflow")
        return False
    
    # Evaluate model
    if not evaluate_model(config):
        logger.error("Model evaluation failed, stopping workflow")
        return False
    
    logger.info("Full workflow completed successfully")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Credit Risk Assessment System")
    
    # General arguments
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    
    # Command selection
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data processing command
    data_parser = subparsers.add_parser("process-data", help="Process and prepare data")
    
    # Model training command
    train_parser = subparsers.add_parser("train", help="Train credit risk model")
    
    # Model evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start prediction API")
    
    # Monitoring command
    monitor_parser = subparsers.add_parser("monitor", help="Start monitoring dashboard")
    
    # Drift detection command
    drift_parser = subparsers.add_parser("detect-drift", help="Run drift detection")
    drift_parser.add_argument("--current-data", required=True, help="Path to current data for drift detection")
    
    # Full workflow command
    workflow_parser = subparsers.add_parser("full-workflow", help="Run complete workflow")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['config_path'] = args.config
    
    # Create necessary directories
    create_directories(config)
    
    # Execute command
    if args.command == "process-data":
        process_data(config)
    elif args.command == "train":
        train_model(config)
    elif args.command == "evaluate":
        evaluate_model(config)
    elif args.command == "api":
        start_api(config)
    elif args.command == "monitor":
        start_monitoring(config)
    elif args.command == "detect-drift":
        # Update config with current data path
        config.setdefault('drift_detection', {})['current_data_path'] = args.current_data
        run_drift_detection(config)
    elif args.command == "full-workflow":
        run_full_workflow(config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()