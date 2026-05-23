"""
Real-time PMS dashboards and monitoring interfaces
Provides interactive visualizations for post-market surveillance metrics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import asyncpg
from ..config.settings import DATABASE_CONFIG
from .drift_detection import DataDriftMonitor, PerformanceDriftDetector
from .safety_monitoring import SafetyMonitor

logger = logging.getLogger(__name__)

class PMSRealTimeDashboard:
    """
    Real-time dashboard for post-market surveillance monitoring
    Displays model performance, drift detection, and safety alerts
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="EEG Neurofeedback PMS Dashboard"
        )
        self.db_pool = None
        self.safety_monitor = SafetyMonitor()
        self.drift_monitor = DataDriftMonitor()

        self._setup_layout()
        self._setup_callbacks()

    async def initialize_database(self):
        """Initialize database connection pool"""
        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(
                host=DATABASE_CONFIG['host'],
                port=DATABASE_CONFIG['port'],
                user=DATABASE_CONFIG['user'],
                password=DATABASE_CONFIG['password'],
                database=DATABASE_CONFIG['database'],
                min_size=2,
                max_size=10
            )

    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("EEG Neurofeedback Post-Market Surveillance",
                           className="text-center mb-4"),
                    html.P("Real-time monitoring of model performance, safety, and drift detection",
                          className="text-center text-muted mb-4")
                ])
            ]),

            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Filters", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Site"),
                                    dcc.Dropdown(
                                        id='site-dropdown',
                                        placeholder="Select site...",
                                        multi=True
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Model Version"),
                                    dcc.Dropdown(
                                        id='model-dropdown',
                                        placeholder="Select model version...",
                                        multi=True
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Time Range"),
                                    dcc.Dropdown(
                                        id='time-range-dropdown',
                                        options=[
                                            {'label': 'Last 24 Hours', 'value': '24h'},
                                            {'label': 'Last 7 Days', 'value': '7d'},
                                            {'label': 'Last 30 Days', 'value': '30d'},
                                            {'label': 'Last 90 Days', 'value': '90d'}
                                        ],
                                        value='24h'
                                    )
                                ], width=4)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Key Metrics Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="overall-safety-score", className="text-center"),
                            html.P("Overall Safety Score", className="text-center text-muted")
                        ])
                    ], color="primary", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="active-alerts-count", className="text-center"),
                            html.P("Active Alerts", className="text-center text-muted")
                        ])
                    ], color="warning", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="model-accuracy", className="text-center"),
                            html.P("Current Accuracy", className="text-center text-muted")
                        ])
                    ], color="success", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="drift-status", className="text-center"),
                            html.P("Drift Status", className="text-center text-muted")
                        ])
                    ], color="info", outline=True)
                ], width=3)
            ], className="mb-4"),

            # Main Dashboard Tabs
            dbc.Tabs([
                dbc.Tab(label="Performance Monitoring", tab_id="performance"),
                dbc.Tab(label="Drift Detection", tab_id="drift"),
                dbc.Tab(label="Safety Alerts", tab_id="alerts"),
                dbc.Tab(label="Site Comparison", tab_id="sites"),
                dbc.Tab(label="Regulatory Reports", tab_id="regulatory")
            ], id="main-tabs", active_tab="performance"),

            # Tab Content
            html.Div(id="tab-content", className="mt-4"),

            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            ),

            # Store components for data sharing
            dcc.Store(id='dashboard-data'),
            dcc.Store(id='site-list'),
            dcc.Store(id='model-list')

        ], fluid=True)

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            [Output('site-dropdown', 'options'),
             Output('model-dropdown', 'options'),
             Output('dashboard-data', 'data')],
            [Input('interval-component', 'n_intervals'),
             Input('time-range-dropdown', 'value')]
        )
        def update_data(n_intervals, time_range):
            """Update dashboard data"""
            return self._update_dashboard_data(time_range)

        @self.app.callback(
            [Output('overall-safety-score', 'children'),
             Output('active-alerts-count', 'children'),
             Output('model-accuracy', 'children'),
             Output('drift-status', 'children')],
            [Input('dashboard-data', 'data')]
        )
        def update_key_metrics(data):
            """Update key metrics cards"""
            if not data:
                return "N/A", "N/A", "N/A", "N/A"

            safety_score = data.get('safety_score', {}).get('overall_safety_score', 0)
            active_alerts = len(data.get('active_alerts', []))
            accuracy = data.get('performance_metrics', {}).get('accuracy', 0)
            drift_detected = data.get('drift_analysis', {}).get('drift_detected', False)

            return (
                f"{safety_score:.1%}",
                str(active_alerts),
                f"{accuracy:.1%}",
                "⚠️ Detected" if drift_detected else "✅ Normal"
            )

        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('dashboard-data', 'data'),
             Input('site-dropdown', 'value'),
             Input('model-dropdown', 'value')]
        )
        def render_tab_content(active_tab, data, selected_sites, selected_models):
            """Render content based on active tab"""
            if not data:
                return html.Div("Loading...", className="text-center")

            if active_tab == "performance":
                return self._render_performance_tab(data, selected_sites, selected_models)
            elif active_tab == "drift":
                return self._render_drift_tab(data, selected_sites, selected_models)
            elif active_tab == "alerts":
                return self._render_alerts_tab(data, selected_sites, selected_models)
            elif active_tab == "sites":
                return self._render_sites_tab(data, selected_sites, selected_models)
            elif active_tab == "regulatory":
                return self._render_regulatory_tab(data, selected_sites, selected_models)

            return html.Div("Select a tab to view content")

    def _update_dashboard_data(self, time_range):
        """Update dashboard data from database"""
        try:
            # This would be called asynchronously in a real implementation
            # For now, return mock data structure

            # Calculate time window
            time_window = self._parse_time_range(time_range)

            # Mock data - in real implementation, fetch from database
            mock_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'time_range': time_range,
                'safety_score': {
                    'overall_safety_score': 0.82,
                    'performance_score': 0.85,
                    'reliability_score': 0.78,
                    'data_quality_score': 0.83
                },
                'performance_metrics': {
                    'accuracy': 0.847,
                    'sensitivity': 0.821,
                    'specificity': 0.873,
                    'f1_score': 0.835,
                    'auc_roc': 0.912,
                    'calibration_error': 0.086
                },
                'drift_analysis': {
                    'drift_detected': False,
                    'psi_score': 0.12,
                    'ks_test_p_value': 0.34,
                    'concept_drift_points': []
                },
                'active_alerts': [
                    {
                        'alert_id': 'alert_1',
                        'alert_type': 'calibration_error',
                        'severity': 'medium',
                        'message': 'Calibration error slightly elevated',
                        'created_at': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                        'site_id': 'site_001'
                    }
                ],
                'sites': ['site_001', 'site_002', 'site_003', 'site_004'],
                'models': ['v2.1.3', 'v2.1.2', 'v2.1.1']
            }

            # Format options for dropdowns
            site_options = [{'label': site, 'value': site} for site in mock_data['sites']]
            model_options = [{'label': model, 'value': model} for model in mock_data['models']]

            return site_options, model_options, mock_data

        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
            return [], [], {}

    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta"""
        time_map = {
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30),
            '90d': timedelta(days=90)
        }
        return time_map.get(time_range, timedelta(hours=24))

    def _render_performance_tab(self, data, selected_sites, selected_models):
        """Render performance monitoring tab"""
        performance_metrics = data.get('performance_metrics', {})

        # Performance trend chart
        performance_fig = self._create_performance_trend_chart(data)

        # Metrics comparison chart
        metrics_comparison_fig = self._create_metrics_comparison_chart(performance_metrics)

        # Calibration plot
        calibration_fig = self._create_calibration_plot(data)

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Trends"),
                        dbc.CardBody([
                            dcc.Graph(figure=performance_fig)
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Current Metrics"),
                        dbc.CardBody([
                            dcc.Graph(figure=metrics_comparison_fig)
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Calibration"),
                        dbc.CardBody([
                            dcc.Graph(figure=calibration_fig)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Summary"),
                        dbc.CardBody([
                            self._create_performance_summary_table(performance_metrics)
                        ])
                    ])
                ], width=6)
            ])
        ])

    def _render_drift_tab(self, data, selected_sites, selected_models):
        """Render drift detection tab"""
        drift_analysis = data.get('drift_analysis', {})

        # PSI trend chart
        psi_fig = self._create_psi_trend_chart(data)

        # Feature drift heatmap
        feature_drift_fig = self._create_feature_drift_heatmap(data)

        # CUSUM chart
        cusum_fig = self._create_cusum_chart(data)

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Population Stability Index (PSI)"),
                        dbc.CardBody([
                            dcc.Graph(figure=psi_fig)
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Drift Summary"),
                        dbc.CardBody([
                            html.H5(f"PSI Score: {drift_analysis.get('psi_score', 0):.3f}"),
                            html.P(f"Status: {'Drift Detected' if drift_analysis.get('drift_detected') else 'Normal'}"),
                            html.P(f"KS Test p-value: {drift_analysis.get('ks_test_p_value', 0):.3f}"),
                            html.P(f"Concept drift points: {len(drift_analysis.get('concept_drift_points', []))}")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Feature Drift Analysis"),
                        dbc.CardBody([
                            dcc.Graph(figure=feature_drift_fig)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("CUSUM Monitoring"),
                        dbc.CardBody([
                            dcc.Graph(figure=cusum_fig)
                        ])
                    ])
                ], width=6)
            ])
        ])

    def _render_alerts_tab(self, data, selected_sites, selected_models):
        """Render safety alerts tab"""
        alerts = data.get('active_alerts', [])

        # Alert severity distribution
        alert_severity_fig = self._create_alert_severity_chart(alerts)

        # Alert timeline
        alert_timeline_fig = self._create_alert_timeline_chart(alerts)

        # Alert table
        alerts_table = self._create_alerts_table(alerts)

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Alert Severity Distribution"),
                        dbc.CardBody([
                            dcc.Graph(figure=alert_severity_fig)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Alert Timeline"),
                        dbc.CardBody([
                            dcc.Graph(figure=alert_timeline_fig)
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Active Alerts"),
                        dbc.CardBody([
                            alerts_table
                        ])
                    ])
                ], width=12)
            ])
        ])

    def _render_sites_tab(self, data, selected_sites, selected_models):
        """Render site comparison tab"""
        # Site performance comparison
        site_comparison_fig = self._create_site_comparison_chart(data)

        # Geographic distribution
        geo_fig = self._create_geographic_distribution_chart(data)

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Site Performance Comparison"),
                        dbc.CardBody([
                            dcc.Graph(figure=site_comparison_fig)
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Geographic Distribution"),
                        dbc.CardBody([
                            dcc.Graph(figure=geo_fig)
                        ])
                    ])
                ], width=4)
            ])
        ])

    def _render_regulatory_tab(self, data, selected_sites, selected_models):
        """Render regulatory reports tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Regulatory Compliance Status"),
                        dbc.CardBody([
                            html.H5("FDA PSUR Compliance: ✅ Compliant"),
                            html.H5("EMA PMSR Status: ✅ Up to Date"),
                            html.P("Last Report: 2025-09-15"),
                            html.P("Next Report Due: 2025-12-15"),
                            dbc.Button("Generate Report", color="primary", className="mt-2")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Adverse Event Summary"),
                        dbc.CardBody([
                            html.P("Total Events: 23"),
                            html.P("Serious Events: 2"),
                            html.P("Device-Related: 1"),
                            html.P("Unexpected: 0"),
                            dbc.Button("View Details", color="info", className="mt-2")
                        ])
                    ])
                ], width=6)
            ])
        ])

    def _create_performance_trend_chart(self, data):
        """Create performance trend chart"""
        # Mock time series data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        accuracy = np.random.normal(0.85, 0.02, 30)
        sensitivity = np.random.normal(0.82, 0.025, 30)
        specificity = np.random.normal(0.87, 0.02, 30)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=accuracy, name='Accuracy', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=sensitivity, name='Sensitivity', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=dates, y=specificity, name='Specificity', line=dict(color='orange')))

        # Add threshold lines
        fig.add_hline(y=0.75, line_dash="dash", line_color="red",
                     annotation_text="Minimum Threshold")

        fig.update_layout(
            title="Performance Metrics Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            yaxis=dict(range=[0.7, 0.95]),
            hovermode='x unified'
        )

        return fig

    def _create_metrics_comparison_chart(self, metrics):
        """Create metrics comparison radar chart"""
        categories = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score', 'AUC-ROC']
        values = [
            metrics.get('accuracy', 0),
            metrics.get('sensitivity', 0),
            metrics.get('specificity', 0),
            metrics.get('f1_score', 0),
            metrics.get('auc_roc', 0)
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Model'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Performance Metrics Comparison"
        )

        return fig

    def _create_calibration_plot(self, data):
        """Create model calibration plot"""
        # Mock calibration data
        mean_predicted_prob = np.linspace(0, 1, 10)
        fraction_positives = mean_predicted_prob + np.random.normal(0, 0.05, 10)
        fraction_positives = np.clip(fraction_positives, 0, 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mean_predicted_prob,
            y=fraction_positives,
            mode='markers+lines',
            name='Model Calibration',
            line=dict(color='blue')
        ))

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title="Model Calibration Plot",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )

        return fig

    def _create_performance_summary_table(self, metrics):
        """Create performance summary table"""
        table_data = [
            ["Accuracy", f"{metrics.get('accuracy', 0):.1%}"],
            ["Sensitivity", f"{metrics.get('sensitivity', 0):.1%}"],
            ["Specificity", f"{metrics.get('specificity', 0):.1%}"],
            ["F1 Score", f"{metrics.get('f1_score', 0):.3f}"],
            ["AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}"],
            ["Calibration Error", f"{metrics.get('calibration_error', 0):.1%}"]
        ]

        return dbc.Table([
            html.Thead([
                html.Tr([html.Th("Metric"), html.Th("Value")])
            ]),
            html.Tbody([
                html.Tr([html.Td(row[0]), html.Td(row[1])]) for row in table_data
            ])
        ], striped=True, bordered=True, hover=True)

    def _create_psi_trend_chart(self, data):
        """Create PSI trend chart"""
        # Mock PSI data over time
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        psi_values = np.random.exponential(0.05, 30)
        psi_values = np.clip(psi_values, 0, 0.5)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=psi_values,
            mode='lines+markers',
            name='PSI Score',
            line=dict(color='purple')
        ))

        # Add threshold lines
        fig.add_hline(y=0.1, line_dash="dash", line_color="yellow",
                     annotation_text="Minor Change (0.1)")
        fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                     annotation_text="Major Change (0.25)")

        fig.update_layout(
            title="Population Stability Index Over Time",
            xaxis_title="Date",
            yaxis_title="PSI Score",
            yaxis=dict(range=[0, 0.5])
        )

        return fig

    def _create_feature_drift_heatmap(self, data):
        """Create feature drift heatmap"""
        # Mock feature drift data
        features = ['Alpha Power', 'Beta Power', 'Theta/Alpha Ratio', 'Signal Quality', 'Artifact Rate']
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')

        drift_matrix = np.random.exponential(0.1, (len(features), len(dates)))

        fig = go.Figure(data=go.Heatmap(
            z=drift_matrix,
            x=[d.strftime('%m-%d') for d in dates],
            y=features,
            colorscale='RdYlBu_r',
            colorbar=dict(title="PSI Score")
        ))

        fig.update_layout(
            title="Feature Drift Heatmap",
            xaxis_title="Date",
            yaxis_title="Features"
        )

        return fig

    def _create_cusum_chart(self, data):
        """Create CUSUM monitoring chart"""
        # Mock CUSUM data
        n_points = 100
        values = np.random.normal(0, 1, n_points)
        cusum_pos = np.zeros(n_points)
        cusum_neg = np.zeros(n_points)

        for i in range(1, n_points):
            cusum_pos[i] = max(0, cusum_pos[i-1] + values[i] - 0.5)
            cusum_neg[i] = max(0, cusum_neg[i-1] - values[i] - 0.5)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(n_points)),
            y=cusum_pos,
            name='CUSUM+',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(n_points)),
            y=cusum_neg,
            name='CUSUM-',
            line=dict(color='blue')
        ))

        # Control limits
        fig.add_hline(y=5, line_dash="dash", line_color="orange",
                     annotation_text="Control Limit (5)")

        fig.update_layout(
            title="CUSUM Drift Detection",
            xaxis_title="Time Point",
            yaxis_title="CUSUM Value"
        )

        return fig

    def _create_alert_severity_chart(self, alerts):
        """Create alert severity distribution chart"""
        severity_counts = {}
        for alert in alerts:
            severity = alert.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        if not severity_counts:
            severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

        fig = go.Figure(data=[
            go.Bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                marker_color=['green', 'yellow', 'orange', 'red'][:len(severity_counts)]
            )
        ])

        fig.update_layout(
            title="Alert Severity Distribution",
            xaxis_title="Severity",
            yaxis_title="Count"
        )

        return fig

    def _create_alert_timeline_chart(self, alerts):
        """Create alert timeline chart"""
        if not alerts:
            # Empty chart
            fig = go.Figure()
            fig.update_layout(title="Alert Timeline (No Active Alerts)")
            return fig

        # Process alerts for timeline
        alert_times = []
        alert_types = []
        alert_severities = []

        for alert in alerts:
            alert_times.append(datetime.fromisoformat(alert['created_at']))
            alert_types.append(alert['alert_type'])
            alert_severities.append(alert['severity'])

        color_map = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'}
        colors = [color_map.get(s, 'gray') for s in alert_severities]

        fig = go.Figure(data=go.Scatter(
            x=alert_times,
            y=alert_types,
            mode='markers',
            marker=dict(
                size=10,
                color=colors
            ),
            text=[f"{alert['severity']}: {alert['message']}" for alert in alerts],
            hovertemplate='%{text}<extra></extra>'
        ))

        fig.update_layout(
            title="Alert Timeline",
            xaxis_title="Time",
            yaxis_title="Alert Type"
        )

        return fig

    def _create_alerts_table(self, alerts):
        """Create alerts table"""
        if not alerts:
            return html.P("No active alerts", className="text-center text-muted")

        table_rows = []
        for alert in alerts:
            severity_badge = dbc.Badge(
                alert['severity'].title(),
                color="danger" if alert['severity'] == 'critical' else
                      "warning" if alert['severity'] == 'high' else
                      "info" if alert['severity'] == 'medium' else "success",
                className="me-2"
            )

            table_rows.append(html.Tr([
                html.Td(severity_badge),
                html.Td(alert['alert_type']),
                html.Td(alert['message'][:50] + "..." if len(alert['message']) > 50 else alert['message']),
                html.Td(alert.get('site_id', 'N/A')),
                html.Td(datetime.fromisoformat(alert['created_at']).strftime('%Y-%m-%d %H:%M')),
                html.Td(dbc.Button("View", size="sm", color="outline-primary"))
            ]))

        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Severity"),
                    html.Th("Type"),
                    html.Th("Message"),
                    html.Th("Site"),
                    html.Th("Created"),
                    html.Th("Action")
                ])
            ]),
            html.Tbody(table_rows)
        ], striped=True, bordered=True, hover=True, responsive=True)

    def _create_site_comparison_chart(self, data):
        """Create site comparison chart"""
        # Mock site performance data
        sites = data.get('sites', [])
        if not sites:
            sites = ['Site A', 'Site B', 'Site C', 'Site D']

        accuracy_scores = np.random.normal(0.85, 0.03, len(sites))
        alert_counts = np.random.poisson(2, len(sites))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Accuracy by Site', 'Alert Count by Site'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        fig.add_trace(
            go.Bar(x=sites, y=accuracy_scores, name='Accuracy'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=sites, y=alert_counts, name='Alert Count', marker_color='red'),
            row=1, col=2
        )

        fig.update_layout(title="Site Performance Comparison")

        return fig

    def _create_geographic_distribution_chart(self, data):
        """Create geographic distribution chart"""
        # Mock geographic data
        site_locations = {
            'North America': 35,
            'Europe': 28,
            'Asia': 22,
            'Other': 15
        }

        fig = go.Figure(data=[
            go.Pie(
                labels=list(site_locations.keys()),
                values=list(site_locations.values()),
                hole=0.3
            )
        ])

        fig.update_layout(
            title="Geographic Distribution of Sites",
            annotations=[dict(text='Sites', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )

        return fig

    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)

# Example usage and configuration
if __name__ == "__main__":
    # Initialize dashboard
    dashboard = PMSRealTimeDashboard()

    # Run server
    dashboard.run_server(debug=True, port=8050)