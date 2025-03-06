# Credit Risk Assessment System

![GitHub](https://img.shields.io/github/license/yourusername/credit-risk-assessment)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/yourusername/credit-risk-assessment)

A production-ready machine learning system for predicting loan default probabilities and managing credit risk.

## 🔍 Overview

This repository contains an end-to-end machine learning pipeline for credit risk assessment. I built this project to tackle the challenge of accurately predicting loan defaults while maintaining model explainability for regulatory compliance. The system combines traditional credit scoring techniques with modern ML approaches and is designed for real-world deployment in financial institutions.

## ✨ Key Features

- **End-to-end ML Pipeline**: From data ingestion to model deployment
- **Model Flexibility**: Multiple algorithms with ensemble capabilities
- **Comprehensive Feature Engineering**: Domain-specific credit risk features
- **API for Real-time Predictions**: Fast, scalable inference
- **Monitoring Dashboard**: Track model performance over time
- **Automated Drift Detection**: Ensure model reliability
- **Model Explainability**: SHAP values and feature importance

## 🛠️ Tech Stack

- **Core**: Python 3.8+, pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **API**: FastAPI
- **Monitoring**: Dash, plotly
- **Testing**: pytest
- **Visualization**: matplotlib, seaborn

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-assessment.git
cd credit-risk-assessment
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create your configuration file by copying the template:
```bash
cp config/config.sample.yaml config/config.yaml
```

5. Edit the configuration to match your environment and data sources.

## 🚀 Usage

### Run the Full Pipeline

The quickest way to see everything in action is to run the complete workflow:

```bash
python main.py full-workflow
```

This command will:
1. Process training and test data (or generate synthetic data if none exists)
2. Engineer features
3. Train a credit risk model (default: gradient boosting)
4. Evaluate model performance
5. Generate visualization reports
6. Save the model for inference

### Individual Components

You can also run individual components of the pipeline:

```bash
# Process data only
python main.py process-data

# Train model
python main.py train

# Evaluate model
python main.py evaluate

# Start API server
python main.py api

# Start monitoring dashboard
python main.py monitor

# Run drift detection
python main.py detect-drift --current-data path/to/new_data.csv
```

## 📁 Project Structure

```
.
├── data/
│   ├── processed/
│   │   ├── processed_test.csv
│   │   └── processed_train.csv
│   └── raw/
├── logs/
├── models/
├── monitoring/
│   ├── dashboard.py
│   └── drift_detection.py
├── reports/
│   ├── figures/
│   ├── model_performance/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquisition.py
│   │   ├── features.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── optimize.py
│   │   ├── predict.py
│   │   └── train.py
│   └── visualization/
│       ├── __init__.py
│       ├── eda.py
│       └── model_analysis.py
├── .gitignore
├── main.py
└── README.md
```

## ⚙️ Configuration

The system uses two main configuration files:

### `config.yaml`

Contains general settings, paths, and data configurations:

```yaml
project:
  name: credit_risk_assessment
  version: 1.0.0

paths:
  raw_data: data/raw/
  processed_data: data/processed/
  models: models/
  reports: reports/

data:
  train_file: credit_data_train.csv
  test_file: credit_data_test.csv
  target: default_flag
  random_state: 42
  test_size: 0.3
  
features:
  categorical:
    - employment_status
    - housing_status
    - product_type
    - purpose
  numerical:
    - loan_amount
    - interest_rate
    - income
    - debt_to_income_ratio
    - credit_score
```

### `model_config.yaml`

Contains model-specific hyperparameters:

```yaml
gradient_boosting:
  n_estimators: 200
  learning_rate: 0.05
  max_depth: 4
  subsample: 0.8
  
logistic_regression:
  C: 0.1
  class_weight: balanced
  max_iter: 1000
```

## 🌐 API Reference

After starting the API server with `python main.py api`, you can access:

- API documentation: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST123",
    "loan_amount": 25000,
    "interest_rate": 0.042,
    "term": 36,
    "income": 85000,
    "debt_to_income_ratio": 0.32,
    "employment_status": "Employed",
    "housing_status": "Mortgage",
    "credit_score": 720,
    "purpose": "Debt Consolidation"
  }'
```

## 💻 Local Development

For development, you can use the synthetic data generator built into the system:

```bash
# Generate synthetic data for development
python -c "from src.data.acquisition import create_dummy_data; create_dummy_data(10000).to_csv('data/raw/synthetic_credit_data.csv', index=False)"
```

Then run the pipeline using this data by updating your config.yaml.

## 📊 Monitoring Dashboard

The monitoring dashboard provides real-time insights into model performance. Start it with:

```bash
python main.py monitor
```

Visit http://localhost:8050 to access:
- Performance metrics over time
- Population stability index
- Risk tier distribution
- Feature drift detection
- Detailed reports

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Make sure to run tests before submitting:

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Notes

- The system automatically generates synthetic data if no real data is provided
- For production use, replace the synthetic data with real credit history data
- The model is designed to be explainable to comply with regulatory requirements
- Performance metrics in the reports directory help track model quality over time

---

Made with ❤️ by Ajit
