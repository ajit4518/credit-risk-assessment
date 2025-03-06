"""
API application for credit risk assessment.
Provides endpoints for model predictions and explanations.
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import yaml
from datetime import datetime

# Import from src
from src.data.acquisition import load_config
from src.models.predict import (
    predict_default_probability, 
    predict_default_with_threshold,
    calculate_risk_score,
    generate_model_explanations
)
from src.data.preprocessing import create_preprocessing_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Credit Risk Assessment API",
    description="API for assessing credit risk of loan applicants",
    version="1.0.0"
)

# Load configuration
config = load_config('config/config.yaml')
model_path = os.path.join(config['paths']['models'], 'credit_risk_model.pkl')
preprocessor_path = os.path.join(config['paths']['models'], 'preprocessor.pkl')

# Global variables for model and preprocessor
model = None
preprocessor = None

# Pydantic models for request/response
class LoanApplication(BaseModel):
    """Loan application data"""
    customer_id: str = Field(..., description="Unique customer identifier")
    loan_amount: float = Field(..., description="Requested loan amount")
    interest_rate: float = Field(..., description="Proposed interest rate")
    term: int = Field(..., description="Loan term in months")
    income: float = Field(..., description="Annual income")
    debt_to_income_ratio: float = Field(..., description="Debt-to-income ratio")
    employment_status: str = Field(..., description="Employment status")
    housing_status: str = Field(..., description="Housing status")
    credit_score: int = Field(..., description="Credit score")
    purpose: str = Field(..., description="Loan purpose")
    
    # Optional fields
    num_inquiries_12m: Optional[int] = Field(None, description="Number of credit inquiries in last 12 months")
    num_delinquencies_24m: Optional[int] = Field(None, description="Number of delinquencies in last 24 months")
    oldest_credit_line_age: Optional[int] = Field(None, description="Age of oldest credit line in months")
    total_credit_lines: Optional[int] = Field(None, description="Total number of credit lines")

class PredictionResponse(BaseModel):
    """Prediction response"""
    customer_id: str
    default_probability: float
    risk_score: int
    risk_tier: str
    approval_recommendation: str
    request_timestamp: str

class ExplanationResponse(BaseModel):
    """Explanation response"""
    customer_id: str
    default_probability: float
    risk_score: int
    top_factors: List[Dict[str, Any]]
    request_timestamp: str

@app.on_event("startup")
async def startup_event():
    """
    Load model and preprocessor on startup
    """
    global model, preprocessor
    
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Load preprocessor
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {str(e)}")
        
        # If model doesn't exist yet, create a dummy preprocessor
        logger.warning("Creating dummy preprocessor and model for development")
        
        # Get configuration
        categorical_cols = config['features']['categorical']
        numerical_cols = config['features']['numerical']
        
        # Create dummy preprocessor
        preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
        
        # Create dummy model (logistic regression)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Credit Risk Assessment API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "/predict",
            "/explain",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    status = "healthy" if model is not None and preprocessor is not None else "unhealthy"
    return {
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """
    Predict default probability for a loan application
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert application to DataFrame
        data = pd.DataFrame([application.dict()])
        
        # Add origination date (current date)
        data['origination_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Fill missing values with defaults
        for col in ['num_inquiries_12m', 'num_delinquencies_24m', 'oldest_credit_line_age', 'total_credit_lines']:
            if col in data.columns and data[col].isnull().any():
                data[col] = 0
        
        # Preprocess data (if preprocessor expects these operations to be done)
        # In practice, this might need more complex logic
        
        # Make prediction
        default_prob = predict_default_probability(model, data)[0]
        
        # Calculate risk score (invert probability, higher score = lower risk)
        risk_score = int(calculate_risk_score(np.array([default_prob]))[0])
        
        # Determine risk tier
        if default_prob < 0.05:
            risk_tier = "Very Low Risk"
        elif default_prob < 0.1:
            risk_tier = "Low Risk"
        elif default_prob < 0.2:
            risk_tier = "Moderate Risk"
        elif default_prob < 0.3:
            risk_tier = "High Risk"
        else:
            risk_tier = "Very High Risk"
        
        # Determine approval recommendation
        if default_prob < 0.1:
            recommendation = "Approve"
        elif default_prob < 0.2:
            recommendation = "Conditionally Approve"
        elif default_prob < 0.3:
            recommendation = "Review"
        else:
            recommendation = "Decline"
        
        # Prepare response
        response = PredictionResponse(
            customer_id=application.customer_id,
            default_probability=round(float(default_prob), 4),
            risk_score=risk_score,
            risk_tier=risk_tier,
            approval_recommendation=recommendation,
            request_timestamp=datetime.now().isoformat()
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplanationResponse)
async def explain(application: LoanApplication):
    """
    Explain prediction for a loan application
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert application to DataFrame
        data = pd.DataFrame([application.dict()])
        
        # Add origination date (current date)
        data['origination_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Fill missing values with defaults
        for col in ['num_inquiries_12m', 'num_delinquencies_24m', 'oldest_credit_line_age', 'total_credit_lines']:
            if col in data.columns and data[col].isnull().any():
                data[col] = 0
        
        # Make prediction
        default_prob = predict_default_probability(model, data)[0]
        
        # Calculate risk score
        risk_score = int(calculate_risk_score(np.array([default_prob]))[0])
        
        # Generate explanations
        try:
            explanations = generate_model_explanations(model, data, 0)
            top_factors = explanations['top_features'][:5]  # Get top 5 factors
        except Exception as ex:
            logger.error(f"Error generating explanations: {str(ex)}")
            top_factors = [
                {"feature": "Explanation not available", "importance": 0.0}
            ]
        
        # Prepare response
        response = ExplanationResponse(
            customer_id=application.customer_id,
            default_probability=round(float(default_prob), 4),
            risk_score=risk_score,
            top_factors=top_factors,
            request_timestamp=datetime.now().isoformat()
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(applications: List[LoanApplication]):
    """
    Batch predict for multiple applications
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert applications to DataFrame
        data = pd.DataFrame([app.dict() for app in applications])
        
        # Add origination date (current date)
        data['origination_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Fill missing values with defaults
        for col in ['num_inquiries_12m', 'num_delinquencies_24m', 'oldest_credit_line_age', 'total_credit_lines']:
            if col in data.columns and data[col].isnull().any():
                data[col] = 0
        
        # Make prediction
        default_probs = predict_default_probability(model, data)
        
        # Calculate risk scores
        risk_scores = calculate_risk_score(default_probs)
        
        # Prepare responses
        results = []
        for i, app in enumerate(applications):
            default_prob = default_probs[i]
            risk_score = risk_scores[i]
            
            # Determine risk tier
            if default_prob < 0.05:
                risk_tier = "Very Low Risk"
            elif default_prob < 0.1:
                risk_tier = "Low Risk"
            elif default_prob < 0.2:
                risk_tier = "Moderate Risk"
            elif default_prob < 0.3:
                risk_tier = "High Risk"
            else:
                risk_tier = "Very High Risk"
            
            # Determine approval recommendation
            if default_prob < 0.1:
                recommendation = "Approve"
            elif default_prob < 0.2:
                recommendation = "Conditionally Approve"
            elif default_prob < 0.3:
                recommendation = "Review"
            else:
                recommendation = "Decline"
            
            results.append({
                "customer_id": app.customer_id,
                "default_probability": round(float(default_prob), 4),
                "risk_score": int(risk_score),
                "risk_tier": risk_tier,
                "approval_recommendation": recommendation
            })
        
        return {
            "results": results,
            "count": len(results),
            "request_timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))