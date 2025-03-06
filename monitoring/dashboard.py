"""
Monitoring dashboard for credit risk assessment.
Dashboard for tracking model performance and data drift.
"""
import pandas as pd
import numpy as np
import os
import glob
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """Class for creating and running a model monitoring dashboard"""
    
    def __init__(self, 
                drift_reports_dir: str,
                predictions_dir: str,
                config_path: str):
        """
        Initialize monitoring dashboard
        
        Args:
            drift_reports_dir: Directory containing drift reports
            predictions_dir: Directory containing prediction logs
            config_path: Path to configuration file
        """
        self.drift_reports_dir = drift_reports_dir
        self.predictions_dir = predictions_dir
        self.config_path = config_path
        
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'])
        self.app.title = "Credit Risk Model Monitoring"
        
        # Create layout
        self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info("Monitoring dashboard initialized")

    def _load_drift_reports(self) -> pd.DataFrame:
        """
        Load drift reports from files
        
        Returns:
            DataFrame with drift report data
        """
        # Find all drift report files
        report_files = glob.glob(os.path.join(self.drift_reports_dir, "drift_report_*.json"))
        
        if not report_files:
            logger.warning("No drift reports found")
            return pd.DataFrame()
        
        # Load each report
        reports = []
        
        for file in sorted(report_files):
            try:
                with open(file, 'r') as f:
                    report = json.load(f)
                
                # Extract basic info
                report_summary = {
                    'timestamp': report.get('timestamp'),
                    'n_current': report.get('n_current'),
                    'prediction_psi': report.get('prediction_psi'),
                    'prediction_drift_detected': report.get('prediction_drift_detected'),
                    'any_feature_drift_detected': report.get('any_feature_drift_detected')
                }
                
                # Add performance metrics if available
                perf = report.get('performance_metrics', {})
                for metric, value in perf.items():
                    report_summary[f'performance_{metric}'] = value
                
                # Add prediction stats
                pred_stats = report.get('current_prediction_stats', {})
                for key, value in pred_stats.items():
                    report_summary[f'pred_{key}'] = value
                
                reports.append(report_summary)
            except Exception as e:
                logger.error(f"Error loading drift report {file}: {str(e)}")
        
        # Convert to DataFrame
        if reports:
            df = pd.DataFrame(reports)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        else:
            return pd.DataFrame()

    def _load_prediction_logs(self) -> pd.DataFrame:
        """
        Load prediction logs from files
        
        Returns:
            DataFrame with prediction log data
        """
        # Find all prediction log files
        log_files = glob.glob(os.path.join(self.predictions_dir, "predictions_*.csv"))
        
        if not log_files:
            logger.warning("No prediction logs found")
            return pd.DataFrame()
        
        # Load and concatenate logs
        logs = []
        
        for file in sorted(log_files):
            try:
                log = pd.read_csv(file)
                logs.append(log)
            except Exception as e:
                logger.error(f"Error loading prediction log {file}: {str(e)}")
        
        # Concatenate logs
        if logs:
            df = pd.concat(logs, ignore_index=True)
            
            # Convert timestamp to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        else:
            return pd.DataFrame()

    def _create_layout(self):
        """Create dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Credit Risk Model Monitoring Dashboard", className="display-4"),
                html.P("Track model performance, data drift, and prediction patterns over time", className="lead"),
                html.Hr()
            ], className="container mt-4"),
            
            # Date range selector
            html.Div([
                html.H4("Select Date Range"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    display_format='YYYY-MM-DD'
                )
            ], className="container mb-4"),
            
            # Summary cards
            html.Div([
                html.Div([
                    # Top row of cards
                    html.Div([
                        # PSI Card
                        html.Div([
                            html.Div([
                                html.H5("Population Stability Index (PSI)", className="card-title"),
                                html.H2(id="psi-value", className="display-5"),
                                html.P(id="psi-status", className="card-text")
                            ], className="card-body")
                        ], className="card bg-light mb-3 shadow-sm"),
                        
                        # AUC Card
                        html.Div([
                            html.Div([
                                html.H5("ROC AUC Score", className="card-title"),
                                html.H2(id="auc-value", className="display-5"),
                                html.P(id="auc-status", className="card-text")
                            ], className="card-body")
                        ], className="card bg-light mb-3 shadow-sm"),
                        
                        # Default Rate Card
                        html.Div([
                            html.Div([
                                html.H5("Average Predicted Default Rate", className="card-title"),
                                html.H2(id="default-rate-value", className="display-5"),
                                html.P(id="default-rate-change", className="card-text")
                            ], className="card-body")
                        ], className="card bg-light mb-3 shadow-sm")
                    ], className="row row-cols-1 row-cols-md-3 g-4 mb-4")
                ], className="container")
            ]),
            
            # Charts - first row
            html.Div([
                html.Div([
                    # PSI Over Time
                    html.Div([
                        html.Div([
                            html.H5("Population Stability Index Over Time", className="card-title"),
                            dcc.Graph(id="psi-time-chart")
                        ], className="card-body")
                    ], className="card shadow-sm mb-4")
                ], className="col-md-6"),
                
                html.Div([
                    # Performance Metrics Over Time
                    html.Div([
                        html.Div([
                            html.H5("Performance Metrics Over Time", className="card-title"),
                            dcc.Graph(id="performance-time-chart")
                        ], className="card-body")
                    ], className="card shadow-sm mb-4")
                ], className="col-md-6"),
                
                html.Div([
                    # Performance Metrics Over Time
                    html.Div([
                        html.Div([
                            html.H5("Performance Metrics Over Time", className="card-title"),
                            dcc.Graph(id="performance-time-chart")
                        ], className="card-body")
                    ], className="card shadow-sm mb-4")
                ], className="col-md-6")
            ], className="row container"),
            
            # Charts - second row
            html.Div([
                html.Div([
                    # Risk Tier Distribution
                    html.Div([
                        html.Div([
                            html.H5("Risk Tier Distribution Over Time", className="card-title"),
                            dcc.Graph(id="risk-distribution-chart")
                        ], className="card-body")
                    ], className="card shadow-sm mb-4")
                ], className="col-md-6"),
                
                html.Div([
                    # Feature Drift
                    html.Div([
                        html.Div([
                            html.H5("Feature Drift Status", className="card-title"),
                            dcc.Graph(id="feature-drift-chart")
                        ], className="card-body")
                    ], className="card shadow-sm mb-4")
                ], className="col-md-6")
            ], className="row container"),
            
            # Data tables section
            html.Div([
                html.H4("Detailed Reports", className="mb-3"),
                
                # Tabs for different tables
                dcc.Tabs([
                    dcc.Tab(label="Drift Reports", children=[
                        html.Div([
                            html.Div(id="drift-reports-table", className="mt-4")
                        ])
                    ]),
                    dcc.Tab(label="Recent Predictions", children=[
                        html.Div([
                            html.Div(id="predictions-table", className="mt-4")
                        ])
                    ])
                ])
            ], className="container mb-5 mt-4"),
            
            # Footer
            html.Footer([
                html.P("Credit Risk Assessment Monitoring Dashboard - Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                html.P("Refresh the page to update with latest data")
            ], className="container text-center text-muted mt-5 pt-3 border-top")
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [
                Output("psi-value", "children"),
                Output("psi-status", "children"),
                Output("psi-status", "className"),
                Output("auc-value", "children"),
                Output("auc-status", "children"),
                Output("auc-status", "className"),
                Output("default-rate-value", "children"),
                Output("default-rate-change", "children"),
                Output("psi-time-chart", "figure"),
                Output("performance-time-chart", "figure"),
                Output("risk-distribution-chart", "figure"),
                Output("feature-drift-chart", "figure"),
                Output("drift-reports-table", "children"),
                Output("predictions-table", "children")
            ],
            [Input("date-range", "start_date"), Input("date-range", "end_date")]
        )
        def update_dashboard(start_date, end_date):
            """
            Update dashboard based on selected date range
            
            Args:
                start_date: Start date for filtering
                end_date: End date for filtering
                
            Returns:
                Updated dashboard components
            """
            # Convert string dates to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date) + timedelta(days=1)  # Include end date
            
            # Load data
            drift_data = self._load_drift_reports()
            prediction_data = self._load_prediction_logs()
            
            # Filter by date range
            if not drift_data.empty and 'timestamp' in drift_data.columns:
                filtered_drift = drift_data[(drift_data['timestamp'] >= start_date) & 
                                          (drift_data['timestamp'] <= end_date)]
            else:
                filtered_drift = pd.DataFrame()
            
            if not prediction_data.empty and 'timestamp' in prediction_data.columns:
                filtered_predictions = prediction_data[(prediction_data['timestamp'] >= start_date) & 
                                                     (prediction_data['timestamp'] <= end_date)]
            else:
                filtered_predictions = pd.DataFrame()
            
            # Generate outputs
            
            # 1. PSI Value and Status
            if not filtered_drift.empty and 'prediction_psi' in filtered_drift.columns:
                latest_psi = filtered_drift['prediction_psi'].iloc[-1]
                psi_value = f"{latest_psi:.3f}"
                
                if latest_psi > 0.25:
                    psi_status = "High drift detected - retraining recommended"
                    psi_class = "card-text text-danger"
                elif latest_psi > 0.1:
                    psi_status = "Medium drift - monitor closely"
                    psi_class = "card-text text-warning"
                else:
                    psi_status = "Low drift - model stable"
                    psi_class = "card-text text-success"
            else:
                psi_value = "N/A"
                psi_status = "No data available"
                psi_class = "card-text text-muted"
            
            # 2. AUC Value and Status
            if not filtered_drift.empty and 'performance_auc' in filtered_drift.columns:
                latest_auc = filtered_drift['performance_auc'].iloc[-1]
                auc_value = f"{latest_auc:.3f}"
                
                if latest_auc > 0.8:
                    auc_status = "Excellent performance"
                    auc_class = "card-text text-success"
                elif latest_auc > 0.7:
                    auc_status = "Good performance"
                    auc_class = "card-text text-success"
                elif latest_auc > 0.6:
                    auc_status = "Fair performance - consider improvements"
                    auc_class = "card-text text-warning"
                else:
                    auc_status = "Poor performance - retraining needed"
                    auc_class = "card-text text-danger"
            else:
                auc_value = "N/A"
                auc_status = "No performance data available"
                auc_class = "card-text text-muted"
            
            # 3. Default Rate Value and Change
            if not filtered_drift.empty and 'pred_mean' in filtered_drift.columns:
                latest_default_rate = filtered_drift['pred_mean'].iloc[-1]
                default_rate_value = f"{latest_default_rate:.2%}"
                
                if len(filtered_drift) > 1:
                    previous_default_rate = filtered_drift['pred_mean'].iloc[-2]
                    change = latest_default_rate - previous_default_rate
                    if abs(change) < 0.0001:
                        default_rate_change = "No change from previous period"
                    else:
                        change_pct = (change / previous_default_rate) * 100
                        default_rate_change = f"{'↑' if change > 0 else '↓'} {abs(change_pct):.1f}% from previous period"
                else:
                    default_rate_change = "No previous data for comparison"
            else:
                default_rate_value = "N/A"
                default_rate_change = "No data available"
            
            # 4. PSI Time Chart
            if not filtered_drift.empty and 'timestamp' in filtered_drift.columns and 'prediction_psi' in filtered_drift.columns:
                psi_fig = px.line(filtered_drift, x='timestamp', y='prediction_psi', 
                                 title='Population Stability Index (PSI) Over Time')
                
                # Add threshold lines
                psi_fig.add_shape(
                    type='line',
                    x0=filtered_drift['timestamp'].min(),
                    x1=filtered_drift['timestamp'].max(),
                    y0=0.25,
                    y1=0.25,
                    line=dict(color='red', dash='dash'),
                    name='High Drift Threshold'
                )
                
                psi_fig.add_shape(
                    type='line',
                    x0=filtered_drift['timestamp'].min(),
                    x1=filtered_drift['timestamp'].max(),
                    y0=0.1,
                    y1=0.1,
                    line=dict(color='orange', dash='dash'),
                    name='Medium Drift Threshold'
                )
                
                psi_fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='PSI Value',
                    hovermode='x unified'
                )
            else:
                psi_fig = go.Figure()
                psi_fig.update_layout(
                    title='No PSI Data Available',
                    xaxis_title='Date',
                    yaxis_title='PSI Value'
                )
            
            # 5. Performance Time Chart
            if not filtered_drift.empty and 'timestamp' in filtered_drift.columns:
                perf_metrics = [col for col in filtered_drift.columns if col.startswith('performance_')]
                
                if perf_metrics:
                    perf_fig = go.Figure()
                    
                    for metric in perf_metrics:
                        display_name = metric.replace('performance_', '').upper()
                        perf_fig.add_trace(go.Scatter(
                            x=filtered_drift['timestamp'],
                            y=filtered_drift[metric],
                            mode='lines+markers',
                            name=display_name
                        ))
                    
                    perf_fig.update_layout(
                        title='Model Performance Metrics Over Time',
                        xaxis_title='Date',
                        yaxis_title='Metric Value',
                        hovermode='x unified',
                        legend=dict(orientation='h')
                    )
                else:
                    perf_fig = go.Figure()
                    perf_fig.update_layout(
                        title='No Performance Data Available',
                        xaxis_title='Date',
                        yaxis_title='Metric Value'
                    )
            else:
                perf_fig = go.Figure()
                perf_fig.update_layout(
                    title='No Performance Data Available',
                    xaxis_title='Date',
                    yaxis_title='Metric Value'
                )
            
            # 6. Risk Distribution Chart
            if not filtered_drift.empty and 'timestamp' in filtered_drift.columns:
                risk_cols = [col for col in filtered_drift.columns if 
                           col in ['pred_very_low_risk', 'pred_low_risk', 
                                  'pred_moderate_risk', 'pred_high_risk', 'pred_very_high_risk']]
                
                if risk_cols:
                    # Prepare data for stacked area chart
                    risk_data = filtered_drift[['timestamp'] + risk_cols].copy()
                    
                    # Convert to long format for plotly
                    risk_data_long = pd.melt(
                        risk_data,
                        id_vars=['timestamp'],
                        value_vars=risk_cols,
                        var_name='risk_tier',
                        value_name='proportion'
                    )
                    
                    # Clean up risk tier names
                    risk_data_long['risk_tier'] = risk_data_long['risk_tier'].str.replace('pred_', '').str.replace('_', ' ').str.title()
                    
                    # Create stacked area chart
                    risk_fig = px.area(
                        risk_data_long,
                        x='timestamp',
                        y='proportion',
                        color='risk_tier',
                        title='Risk Tier Distribution Over Time',
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    
                    risk_fig.update_layout(
                        xaxis_title='Date',
                        yaxis_title='Proportion',
                        hovermode='x unified',
                        legend=dict(title='Risk Tier')
                    )
                else:
                    risk_fig = go.Figure()
                    risk_fig.update_layout(
                        title='No Risk Tier Data Available',
                        xaxis_title='Date',
                        yaxis_title='Proportion'
                    )
            else:
                risk_fig = go.Figure()
                risk_fig.update_layout(
                    title='No Risk Tier Data Available',
                    xaxis_title='Date',
                    yaxis_title='Proportion'
                )
            
            # 7. Feature Drift Chart
            # This requires deeper processing of drift reports, so we'll just create a placeholder
            feature_fig = go.Figure()
            feature_fig.update_layout(
                title='Feature Drift Status (Detailed reports available in files)',
                xaxis_title='Feature',
                yaxis_title='Drift Status'
            )
            
            # 8. Drift Reports Table
            if not filtered_drift.empty:
                from dash import dash_table
                
                # Select and format columns for display
                display_cols = ['timestamp', 'prediction_psi', 'prediction_drift_detected', 'any_feature_drift_detected']
                perf_cols = [col for col in filtered_drift.columns if col.startswith('performance_')]
                display_cols.extend(perf_cols)
                
                display_data = filtered_drift[display_cols].copy()
                
                # Format timestamps
                if 'timestamp' in display_data.columns:
                    display_data['timestamp'] = display_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Create DataTable
                drift_table = dash_table.DataTable(
                    data=display_data.tail(20).to_dict('records'),
                    columns=[{'name': col.replace('_', ' ').title(), 'id': col} for col in display_data.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    page_size=10
                )
            else:
                drift_table = html.Div("No drift data available for selected period")
            
            # 9. Predictions Table
            if not filtered_predictions.empty:
                from dash import dash_table
                
                # Limit columns if there are too many
                display_cols = filtered_predictions.columns[:10].tolist()
                display_data = filtered_predictions[display_cols].copy()
                
                # Format timestamps
                if 'timestamp' in display_data.columns:
                    display_data['timestamp'] = pd.to_datetime(display_data['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Create DataTable
                predictions_table = dash_table.DataTable(
                    data=display_data.tail(20).to_dict('records'),
                    columns=[{'name': col.replace('_', ' ').title(), 'id': col} for col in display_data.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    page_size=10
                )
            else:
                predictions_table = html.Div("No prediction data available for selected period")
            
            # Return all updated components
            return (
                psi_value, psi_status, psi_class,
                auc_value, auc_status, auc_class,
                default_rate_value, default_rate_change,
                psi_fig, perf_fig, risk_fig, feature_fig,
                drift_table, predictions_table
            )

    def run(self, host: str = '0.0.0.0', port: int = 8050, debug: bool = True):
        """
        Run the dashboard
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode flag
        """
        logger.info(f"Starting monitoring dashboard on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)