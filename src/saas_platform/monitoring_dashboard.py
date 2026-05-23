"""
SaaS Platform Monitoring Dashboard
EEG Neurofeedback Adaptive AI Platform

Real-time monitoring and analytics dashboard for multi-tenant SaaS platform
with comprehensive metrics, alerting, and SLA monitoring.

Key Features:
- Real-time tier performance monitoring and comparison
- SLA compliance tracking with automated alerting
- Customer usage analytics and billing validation
- Security event monitoring and incident response
- Cost optimization and resource utilization analysis
- Clinical outcome tracking and quality metrics

Technologies:
- Streamlit for interactive web dashboard
- Plotly for real-time charts and visualizations
- PostgreSQL + TimescaleDB for metrics storage
- Prometheus for metrics collection
- Grafana for detailed operational dashboards

Usage:
    streamlit run monitoring_dashboard.py --server.port 8501
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import asyncpg
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
import time

# Configure Streamlit page
st.set_page_config(
    page_title="ENAAP SaaS Monitoring",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class MetricData:
    """Structure for metric data points"""
    timestamp: datetime
    value: float
    tier: str
    customer_id: str
    metric_type: str

@dataclass
class SLAMetrics:
    """SLA compliance metrics"""
    uptime_percentage: float
    response_time_p95: float
    error_rate_percentage: float
    availability_target: float
    response_time_target: float
    error_rate_target: float

class SaaSMonitoringDashboard:
    """
    Real-time monitoring dashboard for ENAAP SaaS platform

    Provides comprehensive monitoring across all deployment tiers
    with real-time metrics, alerting, and analytics.
    """

    def __init__(self):
        self.db_pool = None
        self.prometheus_url = "http://prometheus:9090"
        self.grafana_url = "http://grafana:3000"

    async def initialize_db_connection(self):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                database="enaap_monitoring",
                user="monitoring_user",
                password="monitoring_password",
                min_size=5,
                max_size=20
            )
        except Exception as e:
            st.error(f"Database connection failed: {e}")

    def render_dashboard(self):
        """Render the main monitoring dashboard"""
        st.title("🧠 ENAAP SaaS Platform Monitoring")
        st.markdown("Real-time monitoring and analytics for EEG Neurofeedback Adaptive AI Platform")

        # Sidebar controls
        self.render_sidebar()

        # Main dashboard content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Overview", "📊 Performance", "💰 Revenue", "🔒 Security", "🏥 Clinical"
        ])

        with tab1:
            self.render_overview_tab()

        with tab2:
            self.render_performance_tab()

        with tab3:
            self.render_revenue_tab()

        with tab4:
            self.render_security_tab()

        with tab5:
            self.render_clinical_tab()

    def render_sidebar(self):
        """Render sidebar controls and filters"""
        st.sidebar.header("🎛️ Dashboard Controls")

        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            index=2
        )

        # Tier filter
        selected_tiers = st.sidebar.multiselect(
            "Deployment Tiers",
            ["Essential", "Professional", "Enterprise"],
            default=["Essential", "Professional", "Enterprise"]
        )

        # Customer filter
        customer_filter = st.sidebar.text_input("Customer ID Filter (optional)")

        # Refresh controls
        st.sidebar.header("🔄 Refresh Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 300, 30)

        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()

        # System status
        st.sidebar.header("🚦 System Status")
        self.render_system_status()

        return {
            "time_range": time_range,
            "selected_tiers": selected_tiers,
            "customer_filter": customer_filter,
            "auto_refresh": auto_refresh,
            "refresh_interval": refresh_interval
        }

    def render_system_status(self):
        """Render overall system status indicators"""
        # Mock data - replace with real metrics
        status_data = {
            "Overall Health": "🟢 Healthy",
            "Active Customers": "147",
            "Total Sessions": "2,847",
            "Current Incidents": "0"
        }

        for label, value in status_data.items():
            st.sidebar.metric(label, value)

    def render_overview_tab(self):
        """Render overview dashboard tab"""
        st.header("📈 Platform Overview")

        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Revenue (MTD)",
                "$1.2M",
                delta="$45K (+3.8%)",
                delta_color="normal"
            )

        with col2:
            st.metric(
                "Active Customers",
                "147",
                delta="8 (+5.8%)",
                delta_color="normal"
            )

        with col3:
            st.metric(
                "Platform Uptime",
                "99.97%",
                delta="0.02%",
                delta_color="normal"
            )

        with col4:
            st.metric(
                "Avg Response Time",
                "245ms",
                delta="-15ms (-5.8%)",
                delta_color="inverse"
            )

        with col5:
            st.metric(
                "Clinical Sessions",
                "2,847",
                delta="234 (+8.9%)",
                delta_color="normal"
            )

        # Tier performance comparison
        st.subheader("🏆 Tier Performance Comparison")
        tier_comparison = self.generate_tier_comparison_chart()
        st.plotly_chart(tier_comparison, use_container_width=True)

        # Recent alerts and incidents
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🚨 Recent Alerts")
            alerts_df = self.get_recent_alerts()
            st.dataframe(alerts_df, use_container_width=True)

        with col2:
            st.subheader("📊 SLA Status")
            sla_chart = self.generate_sla_status_chart()
            st.plotly_chart(sla_chart, use_container_width=True)

    def render_performance_tab(self):
        """Render performance monitoring tab"""
        st.header("⚡ Performance Monitoring")

        # Response time trends
        st.subheader("📈 Response Time Trends")
        response_time_chart = self.generate_response_time_chart()
        st.plotly_chart(response_time_chart, use_container_width=True)

        # Resource utilization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💾 CPU Utilization")
            cpu_chart = self.generate_cpu_utilization_chart()
            st.plotly_chart(cpu_chart, use_container_width=True)

        with col2:
            st.subheader("🧠 Memory Utilization")
            memory_chart = self.generate_memory_utilization_chart()
            st.plotly_chart(memory_chart, use_container_width=True)

        # Error rates and availability
        st.subheader("❌ Error Rates by Tier")
        error_rate_chart = self.generate_error_rate_chart()
        st.plotly_chart(error_rate_chart, use_container_width=True)

        # Performance heatmap
        st.subheader("🗺️ Performance Heatmap")
        heatmap = self.generate_performance_heatmap()
        st.plotly_chart(heatmap, use_container_width=True)

    def render_revenue_tab(self):
        """Render revenue and billing tab"""
        st.header("💰 Revenue Analytics")

        # Revenue metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Monthly Recurring Revenue", "$1,247,500", delta="$47,500 (+4.0%)")

        with col2:
            st.metric("Average Customer Value", "$8,486", delta="$312 (+3.8%)")

        with col3:
            st.metric("Churn Rate", "2.4%", delta="-0.3% (-11.1%)", delta_color="inverse")

        with col4:
            st.metric("Net Revenue Retention", "118%", delta="3% (+2.6%)")

        # Revenue trends
        st.subheader("📈 Revenue Trends")
        revenue_chart = self.generate_revenue_chart()
        st.plotly_chart(revenue_chart, use_container_width=True)

        # Customer distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏢 Customer Distribution by Tier")
            customer_dist_chart = self.generate_customer_distribution_chart()
            st.plotly_chart(customer_dist_chart, use_container_width=True)

        with col2:
            st.subheader("💳 Usage vs. Billing Validation")
            billing_validation_chart = self.generate_billing_validation_chart()
            st.plotly_chart(billing_validation_chart, use_container_width=True)

        # Top customers table
        st.subheader("🌟 Top Customers by Revenue")
        top_customers_df = self.get_top_customers()
        st.dataframe(top_customers_df, use_container_width=True)

    def render_security_tab(self):
        """Render security monitoring tab"""
        st.header("🔒 Security Monitoring")

        # Security metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Security Events", "14", delta="2 (+16.7%)", delta_color="inverse")

        with col2:
            st.metric("Failed Logins", "23", delta="-5 (-17.9%)", delta_color="normal")

        with col3:
            st.metric("Vulnerability Score", "A-", delta="B+ → A-")

        with col4:
            st.metric("Compliance Score", "98.5%", delta="0.5% (+0.5%)")

        # Security events timeline
        st.subheader("🚨 Security Events Timeline")
        security_timeline = self.generate_security_timeline()
        st.plotly_chart(security_timeline, use_container_width=True)

        # Threat assessment
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Threat Categories")
            threat_chart = self.generate_threat_categories_chart()
            st.plotly_chart(threat_chart, use_container_width=True)

        with col2:
            st.subheader("🌍 Geographic Access Patterns")
            geo_chart = self.generate_geographic_access_chart()
            st.plotly_chart(geo_chart, use_container_width=True)

        # Security incidents table
        st.subheader("📋 Recent Security Incidents")
        security_incidents_df = self.get_security_incidents()
        st.dataframe(security_incidents_df, use_container_width=True)

    def render_clinical_tab(self):
        """Render clinical outcomes monitoring tab"""
        st.header("🏥 Clinical Outcomes")

        # Clinical metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Avg UPDRS Improvement", "68%", delta="3% (+4.6%)")

        with col2:
            st.metric("Patient Satisfaction", "4.7/5.0", delta="0.1 (+2.2%)")

        with col3:
            st.metric("Session Completion Rate", "94.2%", delta="1.8% (+1.9%)")

        with col4:
            st.metric("Adverse Events", "0", delta="0 (0%)")

        # Clinical outcomes trends
        st.subheader("📈 Clinical Outcomes Over Time")
        clinical_outcomes_chart = self.generate_clinical_outcomes_chart()
        st.plotly_chart(clinical_outcomes_chart, use_container_width=True)

        # Biomarker analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧬 Biomarker Improvements")
            biomarker_chart = self.generate_biomarker_chart()
            st.plotly_chart(biomarker_chart, use_container_width=True)

        with col2:
            st.subheader("⚖️ Safety Gate Activations")
            safety_gates_chart = self.generate_safety_gates_chart()
            st.plotly_chart(safety_gates_chart, use_container_width=True)

        # Site performance comparison
        st.subheader("🏥 Clinical Site Performance")
        site_performance_df = self.get_site_performance()
        st.dataframe(site_performance_df, use_container_width=True)

    def generate_tier_comparison_chart(self):
        """Generate tier performance comparison chart"""
        # Mock data - replace with real metrics
        tiers = ["Essential", "Professional", "Enterprise"]
        metrics = ["Uptime %", "Response Time", "Satisfaction", "Revenue"]

        data = {
            "Essential": [99.5, 85, 4.2, 78],
            "Professional": [99.9, 92, 4.6, 89],
            "Enterprise": [99.97, 96, 4.8, 94]
        }

        fig = go.Figure()

        for tier in tiers:
            fig.add_trace(go.Scatterpolar(
                r=data[tier],
                theta=metrics,
                fill='toself',
                name=tier
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Tier Performance Comparison"
        )

        return fig

    def generate_response_time_chart(self):
        """Generate response time trends chart"""
        # Generate mock time series data
        dates = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='5T')

        fig = go.Figure()

        for tier in ["Essential", "Professional", "Enterprise"]:
            # Mock data with some realistic patterns
            base_time = {"Essential": 300, "Professional": 200, "Enterprise": 150}[tier]
            noise = np.random.normal(0, 20, len(dates))
            response_times = base_time + noise + 50 * np.sin(np.arange(len(dates)) * 0.1)

            fig.add_trace(go.Scatter(
                x=dates,
                y=response_times,
                mode='lines',
                name=tier,
                line=dict(width=2)
            ))

        fig.update_layout(
            title="Response Time Trends (24 Hours)",
            xaxis_title="Time",
            yaxis_title="Response Time (ms)",
            hovermode='x unified'
        )

        return fig

    def generate_cpu_utilization_chart(self):
        """Generate CPU utilization chart"""
        # Mock data
        tiers = ["Essential", "Professional", "Enterprise"]
        cpu_usage = [45, 62, 71]
        colors = ['green', 'orange', 'red']

        fig = go.Figure(data=[
            go.Bar(x=tiers, y=cpu_usage, marker_color=colors)
        ])

        fig.update_layout(
            title="Current CPU Utilization by Tier",
            yaxis_title="CPU Usage (%)",
            showlegend=False
        )

        # Add threshold line
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                     annotation_text="Warning Threshold")

        return fig

    def generate_memory_utilization_chart(self):
        """Generate memory utilization chart"""
        # Mock data
        tiers = ["Essential", "Professional", "Enterprise"]
        memory_usage = [38, 55, 67]
        colors = ['green', 'orange', 'orange']

        fig = go.Figure(data=[
            go.Bar(x=tiers, y=memory_usage, marker_color=colors)
        ])

        fig.update_layout(
            title="Current Memory Utilization by Tier",
            yaxis_title="Memory Usage (%)",
            showlegend=False
        )

        # Add threshold line
        fig.add_hline(y=75, line_dash="dash", line_color="red",
                     annotation_text="Warning Threshold")

        return fig

    def generate_error_rate_chart(self):
        """Generate error rate chart"""
        dates = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='1H')

        fig = go.Figure()

        for tier in ["Essential", "Professional", "Enterprise"]:
            # Mock error rates
            base_rate = {"Essential": 0.05, "Professional": 0.02, "Enterprise": 0.01}[tier]
            error_rates = np.random.exponential(base_rate, len(dates))

            fig.add_trace(go.Scatter(
                x=dates,
                y=error_rates,
                mode='lines+markers',
                name=tier
            ))

        fig.update_layout(
            title="Error Rates by Tier (24 Hours)",
            xaxis_title="Time",
            yaxis_title="Error Rate (%)",
            hovermode='x unified'
        )

        return fig

    def generate_performance_heatmap(self):
        """Generate performance heatmap"""
        # Mock data for heatmap
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        # Generate mock response time data
        data = np.random.normal(200, 50, (len(days), len(hours)))
        data = np.maximum(data, 50)  # Minimum response time

        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=hours,
            y=days,
            colorscale='RdYlGn_r',
            colorbar=dict(title="Response Time (ms)")
        ))

        fig.update_layout(
            title="Response Time Heatmap (Last 7 Days)",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week"
        )

        return fig

    def generate_sla_status_chart(self):
        """Generate SLA status chart"""
        tiers = ["Essential", "Professional", "Enterprise"]
        sla_targets = [99.5, 99.9, 99.95]
        actual_sla = [99.7, 99.95, 99.97]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=tiers,
            y=sla_targets,
            name='SLA Target',
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            x=tiers,
            y=actual_sla,
            name='Actual Uptime',
            marker_color='darkblue'
        ))

        fig.update_layout(
            title="SLA Compliance by Tier",
            yaxis_title="Uptime (%)",
            barmode='group'
        )

        return fig

    def generate_revenue_chart(self):
        """Generate revenue trends chart"""
        months = pd.date_range(start='2024-01-01', end='2025-01-01', freq='M')

        # Mock revenue data with growth trend
        base_revenue = 800000
        growth_rate = 0.05
        revenues = [base_revenue * (1 + growth_rate) ** i + np.random.normal(0, 20000)
                   for i in range(len(months))]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=months,
            y=revenues,
            mode='lines+markers',
            name='Monthly Revenue',
            line=dict(width=3, color='green')
        ))

        fig.update_layout(
            title="Monthly Recurring Revenue Trend",
            xaxis_title="Month",
            yaxis_title="Revenue ($)",
            yaxis_tickformat='$,.0f'
        )

        return fig

    def generate_customer_distribution_chart(self):
        """Generate customer distribution by tier"""
        tiers = ["Essential", "Professional", "Enterprise"]
        customers = [89, 47, 11]

        fig = go.Figure(data=[
            go.Pie(labels=tiers, values=customers, hole=0.3)
        ])

        fig.update_layout(
            title="Customer Distribution by Tier",
            annotations=[dict(text='Total<br>147', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )

        return fig

    def generate_billing_validation_chart(self):
        """Generate billing validation chart"""
        # Mock data for usage vs billing accuracy
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        accuracy = 98 + np.random.normal(0, 1, len(dates))
        accuracy = np.clip(accuracy, 95, 100)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=accuracy,
            mode='lines+markers',
            name='Billing Accuracy',
            line=dict(color='blue')
        ))

        fig.add_hline(y=99, line_dash="dash", line_color="green",
                     annotation_text="Target Accuracy")

        fig.update_layout(
            title="Billing Accuracy Trend",
            xaxis_title="Date",
            yaxis_title="Accuracy (%)",
            yaxis_range=[94, 101]
        )

        return fig

    def generate_security_timeline(self):
        """Generate security events timeline"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')

        # Mock security events
        events = np.random.poisson(0.1, len(dates))
        severity = np.random.choice(['Low', 'Medium', 'High'], len(dates), p=[0.7, 0.25, 0.05])

        fig = go.Figure()

        for sev in ['Low', 'Medium', 'High']:
            mask = severity == sev
            color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}[sev]

            fig.add_trace(go.Scatter(
                x=dates[mask],
                y=events[mask],
                mode='markers',
                name=f'{sev} Severity',
                marker=dict(color=color, size=8)
            ))

        fig.update_layout(
            title="Security Events Timeline (7 Days)",
            xaxis_title="Time",
            yaxis_title="Number of Events"
        )

        return fig

    def generate_threat_categories_chart(self):
        """Generate threat categories pie chart"""
        categories = ["Brute Force", "DDoS", "Malware", "Phishing", "Other"]
        counts = [12, 8, 3, 2, 5]

        fig = go.Figure(data=[
            go.Pie(labels=categories, values=counts)
        ])

        fig.update_layout(title="Threat Categories (Last 30 Days)")

        return fig

    def generate_geographic_access_chart(self):
        """Generate geographic access patterns"""
        countries = ["United States", "Canada", "United Kingdom", "Germany", "Australia"]
        access_counts = [2847, 456, 234, 189, 123]

        fig = go.Figure(data=[
            go.Bar(x=countries, y=access_counts, marker_color='lightblue')
        ])

        fig.update_layout(
            title="Access Patterns by Country",
            xaxis_title="Country",
            yaxis_title="Access Count"
        )

        return fig

    def generate_clinical_outcomes_chart(self):
        """Generate clinical outcomes trends"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')

        # Mock clinical improvement data
        updrs_improvement = 65 + np.random.normal(0, 3, len(dates))
        satisfaction = 4.5 + np.random.normal(0, 0.2, len(dates))

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=dates, y=updrs_improvement, name="UPDRS Improvement (%)", line=dict(color='blue')),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=dates, y=satisfaction, name="Patient Satisfaction", line=dict(color='green')),
            secondary_y=True,
        )

        fig.update_layout(title="Clinical Outcomes Trends")
        fig.update_xaxis(title_text="Date")
        fig.update_yaxis(title_text="UPDRS Improvement (%)", secondary_y=False)
        fig.update_yaxis(title_text="Patient Satisfaction", secondary_y=True)

        return fig

    def generate_biomarker_chart(self):
        """Generate biomarker improvements chart"""
        biomarkers = ["Alpha/Beta Ratio", "Beta Power", "Coherence", "Burst Duration"]
        improvements = [23, 34, 18, 27]

        fig = go.Figure(data=[
            go.Bar(x=biomarkers, y=improvements, marker_color='lightgreen')
        ])

        fig.update_layout(
            title="Average Biomarker Improvements",
            xaxis_title="Biomarker",
            yaxis_title="Improvement (%)"
        )

        return fig

    def generate_safety_gates_chart(self):
        """Generate safety gates activation chart"""
        gates = ["Quality", "Confidence", "Bounds", "Rate", "Dose", "Vitals"]
        activations = [234, 45, 12, 8, 3, 1]

        fig = go.Figure(data=[
            go.Bar(x=gates, y=activations, marker_color='orange')
        ])

        fig.update_layout(
            title="Safety Gate Activations (Last 30 Days)",
            xaxis_title="Safety Gate",
            yaxis_title="Activations"
        )

        return fig

    def get_recent_alerts(self) -> pd.DataFrame:
        """Get recent alerts data"""
        # Mock alerts data
        alerts_data = {
            "Timestamp": [
                datetime.now() - timedelta(minutes=15),
                datetime.now() - timedelta(hours=2),
                datetime.now() - timedelta(hours=6),
                datetime.now() - timedelta(days=1)
            ],
            "Severity": ["Medium", "Low", "High", "Low"],
            "Alert": [
                "High CPU usage - Professional tier",
                "Failed login attempts detected",
                "Response time SLA breach - Essential",
                "Scheduled maintenance completed"
            ],
            "Status": ["Active", "Resolved", "Resolved", "Resolved"]
        }

        return pd.DataFrame(alerts_data)

    def get_top_customers(self) -> pd.DataFrame:
        """Get top customers by revenue"""
        # Mock customer data
        customers_data = {
            "Customer": ["Mayo Clinic", "Cleveland Clinic", "Johns Hopkins", "Stanford Health", "Mass General"],
            "Tier": ["Enterprise", "Enterprise", "Professional", "Professional", "Enterprise"],
            "Monthly Revenue": ["$25,000", "$25,000", "$8,500", "$8,500", "$25,000"],
            "Utilization": ["87%", "92%", "76%", "83%", "89%"],
            "Satisfaction": [4.9, 4.8, 4.7, 4.6, 4.8]
        }

        return pd.DataFrame(customers_data)

    def get_security_incidents(self) -> pd.DataFrame:
        """Get security incidents data"""
        # Mock security incidents
        incidents_data = {
            "Timestamp": [
                datetime.now() - timedelta(hours=3),
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=2)
            ],
            "Type": ["Failed Login", "DDoS Attempt", "Suspicious API Access"],
            "Severity": ["Low", "Medium", "High"],
            "Source IP": ["192.168.1.100", "10.0.0.15", "203.0.113.45"],
            "Status": ["Mitigated", "Blocked", "Investigated"]
        }

        return pd.DataFrame(incidents_data)

    def get_site_performance(self) -> pd.DataFrame:
        """Get clinical site performance data"""
        # Mock site performance data
        sites_data = {
            "Site": ["Mayo Clinic", "Cleveland Clinic", "Johns Hopkins", "Stanford", "Mass General"],
            "Patients": [45, 38, 29, 33, 41],
            "Avg UPDRS Improvement": ["72%", "69%", "65%", "71%", "68%"],
            "Session Completion": ["96%", "94%", "92%", "95%", "93%"],
            "Satisfaction": [4.8, 4.7, 4.6, 4.7, 4.8],
            "Adverse Events": [0, 0, 1, 0, 0]
        }

        return pd.DataFrame(sites_data)

# Main application
def main():
    """Main application entry point"""
    dashboard = SaaSMonitoringDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()