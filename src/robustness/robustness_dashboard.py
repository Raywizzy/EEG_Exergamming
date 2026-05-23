"""
Robustness Dashboard and Visualization System
Interactive dashboards for stress testing results and regulatory evidence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import base64
import io

logger = logging.getLogger(__name__)

class RobustnessDashboard:
    """Interactive dashboard for robustness testing results"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)

        # Set visualization style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def generate_stress_test_overview(self, campaign_id: str = None,
                                    model_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive stress test overview"""

        with sqlite3.connect(self.db_path) as conn:
            # Base query conditions
            where_conditions = []
            params = []

            if campaign_id:
                # Get tests from specific campaign date range
                cursor = conn.execute("""
                    SELECT started_at, completed_at FROM stress_test_campaigns
                    WHERE campaign_id = ?
                """, (campaign_id,))

                campaign_data = cursor.fetchone()
                if campaign_data:
                    where_conditions.append("timestamp BETWEEN ? AND ?")
                    params.extend([campaign_data[0], campaign_data[1]])

            if model_id:
                where_conditions.append("model_id = ?")
                params.append(model_id)

            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

            # Get stress test results
            query = f"""
                SELECT test_type, passed_thresholds, stressed_metrics,
                       performance_drop, perturbation_params, execution_time
                FROM stress_test_results
                {where_clause}
                ORDER BY timestamp DESC
            """

            cursor = conn.execute(query, params)
            results = cursor.fetchall()

        if not results:
            return {"error": "No stress test data found"}

        # Process results
        test_data = []
        for row in results:
            test_type, passed, metrics_json, drop_json, params_json, exec_time = row

            try:
                metrics = json.loads(metrics_json)
                performance_drop = json.loads(drop_json)
                perturbation_params = json.loads(params_json)

                test_data.append({
                    'test_type': test_type,
                    'passed': passed,
                    'accuracy': metrics.get('accuracy', 0),
                    'confidence': metrics.get('confidence', 0),
                    'accuracy_drop': performance_drop.get('accuracy', 0),
                    'perturbation_params': perturbation_params,
                    'execution_time': exec_time
                })
            except json.JSONDecodeError:
                continue

        # Calculate summary statistics
        df = pd.DataFrame(test_data)

        summary = {
            'total_tests': len(df),
            'passed_tests': df['passed'].sum(),
            'pass_rate': df['passed'].mean(),
            'avg_accuracy': df['accuracy'].mean(),
            'avg_accuracy_drop': df['accuracy_drop'].mean(),
            'max_accuracy_drop': df['accuracy_drop'].max(),
            'avg_execution_time': df['execution_time'].mean()
        }

        # Group by test type
        by_test_type = df.groupby('test_type').agg({
            'passed': ['count', 'sum', 'mean'],
            'accuracy': 'mean',
            'accuracy_drop': ['mean', 'max', 'std'],
            'execution_time': 'mean'
        }).round(3)

        return {
            'summary': summary,
            'by_test_type': by_test_type.to_dict(),
            'raw_data': test_data
        }

    def plot_stress_test_heatmap(self, campaign_id: str = None) -> go.Figure:
        """Create heatmap of stress test results across perturbation types"""

        overview = self.generate_stress_test_overview(campaign_id)
        if 'error' in overview:
            return go.Figure().add_annotation(text="No data available",
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)

        df = pd.DataFrame(overview['raw_data'])

        # Create pivot table for heatmap
        # Extract perturbation intensity for each test type
        df['intensity'] = df.apply(self._extract_perturbation_intensity, axis=1)

        # Create bins for intensity
        df['intensity_bin'] = pd.cut(df['intensity'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        # Pivot table: test_type vs intensity_bin, values = pass_rate
        pivot_data = df.groupby(['test_type', 'intensity_bin'])['passed'].mean().unstack(fill_value=0)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            text=np.round(pivot_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Pass Rate")
        ))

        fig.update_layout(
            title="Stress Test Pass Rate Heatmap",
            xaxis_title="Perturbation Intensity",
            yaxis_title="Test Type",
            height=500
        )

        return fig

    def plot_performance_degradation(self, campaign_id: str = None) -> go.Figure:
        """Plot performance degradation across different stress conditions"""

        overview = self.generate_stress_test_overview(campaign_id)
        if 'error' in overview:
            return go.Figure().add_annotation(text="No data available",
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)

        df = pd.DataFrame(overview['raw_data'])
        df['intensity'] = df.apply(self._extract_perturbation_intensity, axis=1)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Drop vs Intensity', 'Accuracy Drop Distribution',
                          'Pass Rate by Test Type', 'Execution Time vs Complexity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Accuracy drop vs intensity scatter plot
        for test_type in df['test_type'].unique():
            test_data = df[df['test_type'] == test_type]
            fig.add_trace(
                go.Scatter(
                    x=test_data['intensity'],
                    y=test_data['accuracy_drop'],
                    mode='markers',
                    name=test_type,
                    showlegend=True
                ),
                row=1, col=1
            )

        # 2. Accuracy drop distribution
        fig.add_trace(
            go.Histogram(
                x=df['accuracy_drop'],
                nbinsx=20,
                name='Accuracy Drop Distribution',
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Pass rate by test type
        pass_rates = df.groupby('test_type')['passed'].mean()
        fig.add_trace(
            go.Bar(
                x=pass_rates.index,
                y=pass_rates.values,
                name='Pass Rate',
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Execution time vs complexity
        fig.add_trace(
            go.Scatter(
                x=df['intensity'],
                y=df['execution_time'],
                mode='markers',
                name='Execution Time',
                showlegend=False
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Perturbation Intensity", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy Drop", row=1, col=1)

        fig.update_xaxes(title_text="Accuracy Drop", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)

        fig.update_xaxes(title_text="Test Type", row=2, col=1)
        fig.update_yaxes(title_text="Pass Rate", row=2, col=1)

        fig.update_xaxes(title_text="Perturbation Intensity", row=2, col=2)
        fig.update_yaxes(title_text="Execution Time (s)", row=2, col=2)

        fig.update_layout(
            title="Performance Degradation Analysis",
            height=800,
            showlegend=True
        )

        return fig

    def plot_confidence_intervals_dashboard(self, analysis_id: str) -> go.Figure:
        """Create interactive confidence intervals dashboard"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_name, confidence_level, point_estimate,
                       lower_bound, upper_bound, method, sample_size
                FROM confidence_intervals
                WHERE analysis_id = ?
                ORDER BY metric_name, confidence_level
            """, (analysis_id,))

            data = cursor.fetchall()

        if not data:
            return go.Figure().add_annotation(text="No confidence interval data found",
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)

        df = pd.DataFrame(data, columns=[
            'metric_name', 'confidence_level', 'point_estimate',
            'lower_bound', 'upper_bound', 'method', 'sample_size'
        ])

        # Create subplots for each metric
        metrics = df['metric_name'].unique()
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=[f"{metric.replace('_', ' ').title()}" for metric in metrics],
            vertical_spacing=0.08
        )

        colors = px.colors.qualitative.Set1

        for i, metric in enumerate(metrics):
            metric_data = df[df['metric_name'] == metric]

            # Sort by confidence level
            metric_data = metric_data.sort_values('confidence_level')

            # Add confidence interval trace
            fig.add_trace(
                go.Scatter(
                    x=metric_data['confidence_level'],
                    y=metric_data['point_estimate'],
                    mode='markers+lines',
                    name=f'{metric} Point Estimate',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=(i == 0)
                ),
                row=i+1, col=1
            )

            # Add error bars
            fig.add_trace(
                go.Scatter(
                    x=metric_data['confidence_level'].tolist() + metric_data['confidence_level'].tolist()[::-1],
                    y=metric_data['upper_bound'].tolist() + metric_data['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor=f'rgba{colors[i % len(colors)][3:-1]}, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{metric} Confidence Band',
                    showlegend=(i == 0)
                ),
                row=i+1, col=1
            )

            # Update axes
            fig.update_xaxes(title_text="Confidence Level", row=i+1, col=1)
            fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=i+1, col=1)

        fig.update_layout(
            title="Statistical Confidence Intervals",
            height=300 * len(metrics),
            showlegend=True
        )

        return fig

    def plot_cross_device_compatibility(self) -> go.Figure:
        """Plot cross-device compatibility matrix"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT device_a, device_b, compatibility_score, adaptation_difficulty
                FROM device_compatibility
                ORDER BY compatibility_score DESC
            """)

            data = cursor.fetchall()

        if not data:
            return go.Figure().add_annotation(text="No cross-device data found",
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)

        df = pd.DataFrame(data, columns=['device_a', 'device_b', 'compatibility_score', 'difficulty'])

        # Create compatibility matrix
        devices = sorted(set(df['device_a'].tolist() + df['device_b'].tolist()))
        matrix = np.zeros((len(devices), len(devices)))

        device_to_idx = {device: i for i, device in enumerate(devices)}

        for _, row in df.iterrows():
            i = device_to_idx[row['device_a']]
            j = device_to_idx[row['device_b']]
            matrix[i, j] = row['compatibility_score']

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=devices,
            y=devices,
            colorscale='RdYlGn',
            text=np.round(matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Compatibility Score")
        ))

        fig.update_layout(
            title="Cross-Device Compatibility Matrix",
            xaxis_title="Target Device",
            yaxis_title="Source Device",
            height=600
        )

        return fig

    def generate_regulatory_summary_plots(self, analysis_id: str) -> Dict[str, go.Figure]:
        """Generate regulatory-ready summary plots"""

        figures = {}

        # 1. Risk Assessment Overview
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT risk_type, risk_level, risk_score, risk_factors
                FROM risk_assessments
                WHERE analysis_id = ?
            """, (analysis_id,))

            risk_data = cursor.fetchall()

        if risk_data:
            df_risk = pd.DataFrame(risk_data, columns=['risk_type', 'risk_level', 'risk_score', 'risk_factors'])

            # Risk level distribution
            risk_counts = df_risk['risk_level'].value_counts()

            fig_risk = go.Figure(data=[
                go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=['green', 'yellow', 'orange', 'red'][:len(risk_counts)]
                )
            ])

            fig_risk.update_layout(
                title="Risk Assessment Distribution",
                xaxis_title="Risk Level",
                yaxis_title="Number of Assessments"
            )

            figures['risk_assessment'] = fig_risk

        # 2. Statistical Significance Summary
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT test_name, p_value, effect_size, interpretation
                FROM statistical_tests
                WHERE analysis_id = ?
            """, (analysis_id,))

            test_data = cursor.fetchall()

        if test_data:
            df_tests = pd.DataFrame(test_data, columns=['test_name', 'p_value', 'effect_size', 'interpretation'])

            # Significance levels
            df_tests['significance'] = df_tests['p_value'].apply(
                lambda p: 'Highly Significant' if p < 0.001 else
                         'Very Significant' if p < 0.01 else
                         'Significant' if p < 0.05 else
                         'Not Significant'
            )

            significance_counts = df_tests['significance'].value_counts()

            fig_sig = go.Figure(data=[
                go.Pie(
                    labels=significance_counts.index,
                    values=significance_counts.values,
                    hole=0.3
                )
            ])

            fig_sig.update_layout(
                title="Statistical Significance Distribution"
            )

            figures['statistical_significance'] = fig_sig

        # 3. Performance Bounds Visualization
        figures['confidence_intervals'] = self.plot_confidence_intervals_dashboard(analysis_id)

        return figures

    def create_executive_summary_plot(self, campaign_id: str = None) -> go.Figure:
        """Create executive-level summary visualization"""

        overview = self.generate_stress_test_overview(campaign_id)
        if 'error' in overview:
            return go.Figure().add_annotation(text="No data available",
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)

        summary = overview['summary']
        by_test_type = overview['by_test_type']

        # Create comprehensive dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Overall Pass Rate', 'Performance Metrics', 'Test Coverage',
                'Execution Efficiency', 'Risk Indicators', 'Robustness Score'
            ),
            specs=[[{"type": "indicator"}, {"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "indicator"}]]
        )

        # 1. Overall Pass Rate (Gauge)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=summary['pass_rate'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Pass Rate (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )

        # 2. Performance Metrics (Bar)
        metrics = ['avg_accuracy', 'avg_accuracy_drop', 'max_accuracy_drop']
        values = [summary[m] for m in metrics]

        fig.add_trace(
            go.Bar(
                x=[m.replace('_', ' ').title() for m in metrics],
                y=values,
                name='Performance Metrics'
            ),
            row=1, col=2
        )

        # 3. Test Coverage (Pie)
        test_types = list(by_test_type.keys())
        test_counts = [by_test_type[tt]['passed']['count'] for tt in test_types]

        fig.add_trace(
            go.Pie(
                labels=test_types,
                values=test_counts,
                name="Test Coverage"
            ),
            row=1, col=3
        )

        # 4. Execution Efficiency (Scatter)
        efficiency_data = []
        for tt in test_types:
            efficiency_data.append({
                'test_type': tt,
                'execution_time': by_test_type[tt]['execution_time']['mean'],
                'pass_rate': by_test_type[tt]['passed']['mean']
            })

        df_eff = pd.DataFrame(efficiency_data)

        fig.add_trace(
            go.Scatter(
                x=df_eff['execution_time'],
                y=df_eff['pass_rate'],
                mode='markers+text',
                text=df_eff['test_type'],
                textposition="top center",
                name='Efficiency'
            ),
            row=2, col=1
        )

        # 5. Risk Indicators (Bar)
        risk_factors = ['High Degradation', 'Execution Timeout', 'Failure Rate']
        risk_values = [
            summary['max_accuracy_drop'] * 100,
            (summary['avg_execution_time'] > 30) * 100,  # Boolean to percentage
            (1 - summary['pass_rate']) * 100
        ]

        fig.add_trace(
            go.Bar(
                x=risk_factors,
                y=risk_values,
                marker_color=['red' if v > 50 else 'yellow' if v > 20 else 'green' for v in risk_values],
                name='Risk Indicators'
            ),
            row=2, col=2
        )

        # 6. Overall Robustness Score (Gauge)
        robustness_score = (
            summary['pass_rate'] * 0.4 +
            (1 - summary['avg_accuracy_drop']) * 0.3 +
            (summary['avg_accuracy'] > 0.7) * 0.3
        ) * 100

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=robustness_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Robustness Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 60], 'color': "red"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                }
            ),
            row=2, col=3
        )

        fig.update_layout(
            title="Executive Robustness Summary",
            height=800,
            showlegend=False
        )

        return fig

    def _extract_perturbation_intensity(self, row: pd.Series) -> float:
        """Extract perturbation intensity from parameters"""
        params = row['perturbation_params']
        test_type = row['test_type']

        if 'noise' in test_type:
            return params.get('snr_db', 1.0)
        elif 'dropout' in test_type:
            return params.get('dropout_rate', 0.1) * 10  # Scale to similar range
        elif 'duration' in test_type:
            factor = params.get('factor', 1.0)
            return abs(factor - 1.0) * 10  # Distance from normal
        elif 'amplitude' in test_type:
            factor = params.get('factor', 1.0)
            return abs(factor - 1.0) * 10  # Distance from normal
        elif 'filter' in test_type:
            return 5.0  # Fixed intensity for filter tests
        elif 'sampling' in test_type:
            target_rate = params.get('target_rate', 250)
            return abs(target_rate - 250) / 250 * 10  # Normalized difference
        else:
            return 1.0  # Default

    def export_dashboard_html(self, figures: Dict[str, go.Figure],
                            output_path: str, title: str = "Robustness Dashboard") -> str:
        """Export dashboard as standalone HTML"""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-header {{ text-align: center; margin-bottom: 30px; }}
                .plot-container {{ margin: 20px 0; }}
                .plot-title {{ font-size: 18px; font-weight: bold; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>{title}</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """

        for i, (name, fig) in enumerate(figures.items()):
            plot_id = f"plot_{i}"

            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">{name.replace('_', ' ').title()}</div>
                <div id="{plot_id}" style="width:100%;height:600px;"></div>
            </div>
            """

        html_content += """
            <script>
        """

        for i, (name, fig) in enumerate(figures.items()):
            plot_id = f"plot_{i}"
            fig_json = fig.to_json()

            html_content += f"""
                Plotly.newPlot('{plot_id}', {fig_json});
            """

        html_content += """
            </script>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        return output_path

    def generate_static_report_plots(self, analysis_id: str, output_dir: str) -> List[str]:
        """Generate static plots for regulatory submissions"""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        saved_plots = []

        # Set style for publication-quality plots
        plt.style.use('default')
        sns.set_context("paper", font_scale=1.2)

        # 1. Confidence intervals plot
        try:
            fig_ci = self.plot_confidence_intervals_dashboard(analysis_id)
            ci_path = output_dir / f"confidence_intervals_{analysis_id}.png"
            fig_ci.write_image(str(ci_path), width=1200, height=800, scale=2)
            saved_plots.append(str(ci_path))
        except Exception as e:
            logger.warning(f"Failed to save confidence intervals plot: {e}")

        # 2. Performance degradation analysis
        try:
            fig_perf = self.plot_performance_degradation()
            perf_path = output_dir / f"performance_degradation_{analysis_id}.png"
            fig_perf.write_image(str(perf_path), width=1200, height=800, scale=2)
            saved_plots.append(str(perf_path))
        except Exception as e:
            logger.warning(f"Failed to save performance degradation plot: {e}")

        # 3. Cross-device compatibility
        try:
            fig_compat = self.plot_cross_device_compatibility()
            compat_path = output_dir / f"cross_device_compatibility_{analysis_id}.png"
            fig_compat.write_image(str(compat_path), width=1000, height=600, scale=2)
            saved_plots.append(str(compat_path))
        except Exception as e:
            logger.warning(f"Failed to save compatibility plot: {e}")

        # 4. Executive summary
        try:
            fig_exec = self.create_executive_summary_plot()
            exec_path = output_dir / f"executive_summary_{analysis_id}.png"
            fig_exec.write_image(str(exec_path), width=1400, height=800, scale=2)
            saved_plots.append(str(exec_path))
        except Exception as e:
            logger.warning(f"Failed to save executive summary plot: {e}")

        return saved_plots