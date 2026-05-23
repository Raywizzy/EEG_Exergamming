"""
SaaS Revenue and Clinical Enrollment Monitoring System
EEG Neurofeedback Adaptive AI Platform

Real-time monitoring system for SaaS revenue metrics, customer success indicators,
and clinical trial enrollment with predictive analytics and automated reporting.

Key Features:
- Real-time SaaS revenue tracking with tier-based analysis
- Clinical trial enrollment monitoring across multiple sites
- Customer health scoring and churn prediction
- Revenue forecasting and ARR projections
- Clinical outcome tracking and safety monitoring
- Automated alerts for revenue and enrollment milestones

Usage:
    streamlit run saas_clinical_monitor.py --server.port 8504
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

class SaaSTier(Enum):
    """SaaS deployment tiers"""
    ESSENTIAL = "essential"      # $2.5K/month
    PROFESSIONAL = "professional"  # $8.5K/month
    ENTERPRISE = "enterprise"    # $25K/month

class CustomerStatus(Enum):
    """Customer lifecycle status"""
    TRIAL = "trial"
    ACTIVE = "active"
    AT_RISK = "at_risk"
    CHURNED = "churned"
    PAUSED = "paused"

class TrialPhase(Enum):
    """Clinical trial phases"""
    SHADOW = "shadow"     # Observation mode
    ASSIST = "assist"     # Human-supervised mode
    ACTIVE = "active"     # Autonomous mode

@dataclass
class SaaSCustomer:
    """Structure for SaaS customer tracking"""
    customer_id: str
    organization_name: str
    tier: SaaSTier
    status: CustomerStatus
    monthly_recurring_revenue: int
    contract_start_date: date
    contract_end_date: date
    trial_start_date: Optional[date]
    trial_end_date: Optional[date]
    billing_cycle: str  # 'monthly', 'quarterly', 'annual'
    seats_licensed: int
    seats_active: int
    usage_percentage: float
    satisfaction_score: float  # 1.0-5.0
    health_score: int  # 0-100
    churn_risk_score: int  # 0-100
    clinical_site_count: int
    total_sessions_completed: int
    last_login_date: Optional[date]
    support_tickets_open: int
    success_manager: str
    deployment_phase: TrialPhase
    created_date: datetime
    last_updated: datetime

@dataclass
class ClinicalSite:
    """Structure for clinical site tracking"""
    site_id: str
    site_name: str
    location: str
    principal_investigator: str
    site_type: str  # 'academic', 'private_practice', 'hospital_system'
    activation_date: Optional[date]
    status: str  # 'pending', 'active', 'paused', 'completed'
    target_enrollment: int
    current_enrollment: int
    enrollment_rate: float  # patients per week
    trial_phase: TrialPhase
    irb_approval_date: Optional[date]
    equipment_installed: bool
    staff_trained: bool
    first_patient_date: Optional[date]
    last_patient_date: Optional[date]
    completion_rate: float  # session completion percentage
    safety_incidents: int
    data_quality_score: int  # 0-100
    site_coordinator: str
    created_date: datetime

@dataclass
class RevenueMetrics:
    """SaaS revenue metrics"""
    monthly_recurring_revenue: int
    annual_recurring_revenue: int
    total_customers: int
    new_customers_this_month: int
    churned_customers_this_month: int
    net_revenue_retention: float
    gross_revenue_retention: float
    customer_acquisition_cost: int
    customer_lifetime_value: int
    average_revenue_per_user: int
    monthly_growth_rate: float
    churn_rate: float

@dataclass
class ClinicalMetrics:
    """Clinical trial metrics"""
    total_sites_active: int
    total_patients_enrolled: int
    enrollment_rate_weekly: float
    target_enrollment: int
    enrollment_percentage: float
    sites_pending_activation: int
    completion_rate: float
    dropout_rate: float
    safety_incidents: int
    protocol_deviations: int
    data_quality_average: float

class SaaSClinicalMonitor:
    """
    Comprehensive SaaS revenue and clinical enrollment monitoring system

    Provides real-time tracking of business metrics, customer health,
    and clinical trial progress with predictive analytics.
    """

    def __init__(self):
        self.customers = self._initialize_customers()
        self.clinical_sites = self._initialize_clinical_sites()
        self.revenue_metrics = self._calculate_revenue_metrics()
        self.clinical_metrics = self._calculate_clinical_metrics()

    def render_dashboard(self):
        """Render SaaS and clinical monitoring dashboard"""
        st.title("💰 SaaS Revenue & Clinical Enrollment Monitor")
        st.markdown("Real-time tracking of business performance and clinical trial progress")

        # Executive summary
        self.render_executive_summary()

        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💰 Revenue Dashboard", "👥 Customer Success", "🏥 Clinical Enrollment", "📈 Growth Analytics", "🚨 Alerts & Forecasts"
        ])

        with tab1:
            self.render_revenue_dashboard()

        with tab2:
            self.render_customer_success()

        with tab3:
            self.render_clinical_enrollment()

        with tab4:
            self.render_growth_analytics()

        with tab5:
            self.render_alerts_forecasts()

        # Sidebar controls
        self.render_sidebar()

    def render_executive_summary(self):
        """Render executive summary with key metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Monthly Recurring Revenue",
                f"${self.revenue_metrics.monthly_recurring_revenue//1000}K",
                delta=f"+${(self.revenue_metrics.monthly_recurring_revenue * 0.15)//1000}K this month"
            )

        with col2:
            st.metric(
                "Active Customers",
                f"{self.revenue_metrics.total_customers}",
                delta=f"+{self.revenue_metrics.new_customers_this_month} new"
            )

        with col3:
            st.metric(
                "Clinical Sites Active",
                f"{self.clinical_metrics.total_sites_active}",
                delta=f"{self.clinical_metrics.sites_pending_activation} pending"
            )

        with col4:
            st.metric(
                "Patients Enrolled",
                f"{self.clinical_metrics.total_patients_enrolled}",
                delta=f"+{int(self.clinical_metrics.enrollment_rate_weekly)} this week"
            )

        with col5:
            net_retention = self.revenue_metrics.net_revenue_retention
            st.metric(
                "Net Revenue Retention",
                f"{net_retention:.0f}%",
                delta=f"{net_retention-100:.0f}% growth"
            )

        # Health indicators
        st.markdown("---")
        self.render_health_indicators()

    def render_health_indicators(self):
        """Render business and clinical health indicators"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            churn_rate = self.revenue_metrics.churn_rate
            status = "🟢" if churn_rate <= 5 else "🟡" if churn_rate <= 10 else "🔴"
            st.markdown(f"**Customer Churn** {status}")
            st.caption(f"{churn_rate:.1f}% monthly churn rate")

        with col2:
            growth_rate = self.revenue_metrics.monthly_growth_rate
            status = "🟢" if growth_rate >= 15 else "🟡" if growth_rate >= 10 else "🔴"
            st.markdown(f"**Revenue Growth** {status}")
            st.caption(f"{growth_rate:.1f}% monthly growth rate")

        with col3:
            enrollment_rate = self.clinical_metrics.enrollment_percentage
            status = "🟢" if enrollment_rate >= 75 else "🟡" if enrollment_rate >= 50 else "🔴"
            st.markdown(f"**Enrollment Progress** {status}")
            st.caption(f"{enrollment_rate:.0f}% of target enrollment")

        with col4:
            safety_score = 100 - (self.clinical_metrics.safety_incidents * 10)
            status = "🟢" if safety_score >= 90 else "🟡" if safety_score >= 80 else "🔴"
            st.markdown(f"**Safety Score** {status}")
            st.caption(f"{max(0, safety_score)}/100 safety rating")

    def render_revenue_dashboard(self):
        """Render revenue tracking dashboard"""
        st.header("💰 Revenue Performance Dashboard")

        # Revenue overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Annual Recurring Revenue", f"${self.revenue_metrics.annual_recurring_revenue//1000000}M")

        with col2:
            st.metric("Customer LTV", f"${self.revenue_metrics.customer_lifetime_value//1000}K")

        with col3:
            st.metric("Average Revenue Per User", f"${self.revenue_metrics.average_revenue_per_user//1000}K")

        # Revenue growth chart
        st.subheader("📈 Monthly Recurring Revenue Growth")
        mrr_growth_chart = self.create_mrr_growth_chart()
        st.plotly_chart(mrr_growth_chart, use_container_width=True)

        # Revenue breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💼 Revenue by Tier")
            revenue_by_tier = self.create_revenue_by_tier_chart()
            st.plotly_chart(revenue_by_tier, use_container_width=True)

        with col2:
            st.subheader("📊 Customer Distribution")
            customer_distribution = self.create_customer_distribution_chart()
            st.plotly_chart(customer_distribution, use_container_width=True)

        # Cohort analysis
        st.subheader("📅 Revenue Cohort Analysis")
        cohort_analysis = self.create_cohort_analysis_chart()
        st.plotly_chart(cohort_analysis, use_container_width=True)

        # Top customers table
        st.subheader("🌟 Top Revenue Customers")
        top_customers = self.create_top_customers_table()
        st.dataframe(top_customers, use_container_width=True)

    def render_customer_success(self):
        """Render customer success and health monitoring"""
        st.header("👥 Customer Success & Health Monitoring")

        # Customer health overview
        col1, col2, col3 = st.columns(3)

        healthy_customers = len([c for c in self.customers if c.health_score >= 80])
        at_risk_customers = len([c for c in self.customers if c.churn_risk_score >= 70])
        avg_satisfaction = np.mean([c.satisfaction_score for c in self.customers])

        with col1:
            st.metric("Healthy Customers", healthy_customers, delta=f"{(healthy_customers/len(self.customers)*100):.0f}%")

        with col2:
            st.metric("At-Risk Customers", at_risk_customers, delta=f"{(at_risk_customers/len(self.customers)*100):.0f}%")

        with col3:
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5.0", delta="Above target (4.5)")

        # Customer health distribution
        st.subheader("🏥 Customer Health Score Distribution")
        health_distribution = self.create_health_score_distribution()
        st.plotly_chart(health_distribution, use_container_width=True)

        # Usage analytics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Usage Patterns")
            usage_patterns = self.create_usage_patterns_chart()
            st.plotly_chart(usage_patterns, use_container_width=True)

        with col2:
            st.subheader("⚠️ Churn Risk Analysis")
            churn_risk = self.create_churn_risk_chart()
            st.plotly_chart(churn_risk, use_container_width=True)

        # Customer details
        st.subheader("📋 Customer Health Details")
        for customer in self.customers:
            if customer.status == CustomerStatus.AT_RISK or customer.churn_risk_score >= 70:
                with st.expander(f"⚠️ {customer.organization_name} - Health Score: {customer.health_score}/100"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                        **Customer Details:**
                        - **Tier:** {customer.tier.value.title()}
                        - **MRR:** ${customer.monthly_recurring_revenue:,}
                        - **Contract End:** {customer.contract_end_date.strftime('%m/%d/%Y')}
                        - **Usage:** {customer.usage_percentage:.0f}%
                        - **Seats:** {customer.seats_active}/{customer.seats_licensed}
                        """)

                    with col2:
                        st.markdown(f"""
                        **Health Indicators:**
                        - **Health Score:** {customer.health_score}/100
                        - **Churn Risk:** {customer.churn_risk_score}/100
                        - **Satisfaction:** {customer.satisfaction_score:.1f}/5.0
                        - **Last Login:** {customer.last_login_date.strftime('%m/%d/%Y') if customer.last_login_date else 'None'}
                        - **Support Tickets:** {customer.support_tickets_open}
                        """)

                    # Recommended actions
                    if customer.churn_risk_score >= 70:
                        st.error(f"🚨 High churn risk - Immediate intervention required")
                        st.markdown("**Recommended Actions:**")
                        st.markdown("• Schedule executive check-in call within 48 hours")
                        st.markdown("• Review contract renewal terms and potential discounts")
                        st.markdown("• Assess technical implementation challenges")

    def render_clinical_enrollment(self):
        """Render clinical trial enrollment monitoring"""
        st.header("🏥 Clinical Trial Enrollment Monitoring")

        # Enrollment overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Enrolled", self.clinical_metrics.total_patients_enrolled)

        with col2:
            st.metric("Target Enrollment", self.clinical_metrics.target_enrollment)

        with col3:
            st.metric("Completion Rate", f"{self.clinical_metrics.completion_rate:.0f}%")

        with col4:
            st.metric("Weekly Enrollment Rate", f"{self.clinical_metrics.enrollment_rate_weekly:.1f}")

        # Enrollment progress
        st.subheader("📈 Enrollment Progress Over Time")
        enrollment_progress = self.create_enrollment_progress_chart()
        st.plotly_chart(enrollment_progress, use_container_width=True)

        # Site performance
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏥 Site Enrollment Performance")
            site_performance = self.create_site_performance_chart()
            st.plotly_chart(site_performance, use_container_width=True)

        with col2:
            st.subheader("⚡ Enrollment Rate by Phase")
            phase_enrollment = self.create_phase_enrollment_chart()
            st.plotly_chart(phase_enrollment, use_container_width=True)

        # Geographic distribution
        st.subheader("🗺️ Geographic Distribution of Sites")
        geographic_distribution = self.create_geographic_distribution_chart()
        st.plotly_chart(geographic_distribution, use_container_width=True)

        # Site details
        st.subheader("📋 Clinical Site Details")
        site_details = self.create_site_details_table()
        st.dataframe(site_details, use_container_width=True)

        # Trial phase progression
        st.subheader("🔄 Trial Phase Progression")
        for site in self.clinical_sites:
            if site.status == 'active':
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**{site.site_name}**")
                    st.caption(f"{site.location} • PI: {site.principal_investigator}")

                with col2:
                    progress = (site.current_enrollment / site.target_enrollment * 100) if site.target_enrollment > 0 else 0
                    st.metric("Enrollment", f"{site.current_enrollment}/{site.target_enrollment}", delta=f"{progress:.0f}%")

                with col3:
                    phase_color = {"shadow": "🔵", "assist": "🟡", "active": "🟢"}[site.trial_phase.value]
                    st.markdown(f"**Phase:** {phase_color} {site.trial_phase.value.title()}")

    def render_growth_analytics(self):
        """Render growth analytics and projections"""
        st.header("📈 Growth Analytics & Projections")

        # Growth metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Monthly Growth Rate", f"{self.revenue_metrics.monthly_growth_rate:.1f}%")

        with col2:
            cac = self.revenue_metrics.customer_acquisition_cost
            st.metric("Customer Acquisition Cost", f"${cac//1000}K")

        with col3:
            ltv_cac_ratio = self.revenue_metrics.customer_lifetime_value / cac
            st.metric("LTV:CAC Ratio", f"{ltv_cac_ratio:.1f}:1")

        with col4:
            payback_months = cac / (self.revenue_metrics.average_revenue_per_user / 12)
            st.metric("Payback Period", f"{payback_months:.0f} months")

        # Revenue projections
        st.subheader("🔮 Revenue Projections")
        revenue_projections = self.create_revenue_projections_chart()
        st.plotly_chart(revenue_projections, use_container_width=True)

        # Growth drivers analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🚀 Growth Drivers")
            growth_drivers = self.create_growth_drivers_chart()
            st.plotly_chart(growth_drivers, use_container_width=True)

        with col2:
            st.subheader("📊 Unit Economics")
            unit_economics = self.create_unit_economics_chart()
            st.plotly_chart(unit_economics, use_container_width=True)

        # Scenario analysis
        st.subheader("🎯 Scenario Analysis")
        scenario_col1, scenario_col2, scenario_col3 = st.columns(3)

        with scenario_col1:
            st.markdown("**🟢 Optimistic Scenario**")
            st.markdown("• 25% monthly growth rate")
            st.markdown("• 5% monthly churn rate")
            st.markdown("• 120% net revenue retention")
            st.metric("12-Month ARR", "$18M", delta="+200%")

        with scenario_col2:
            st.markdown("**🟡 Base Case Scenario**")
            st.markdown("• 15% monthly growth rate")
            st.markdown("• 8% monthly churn rate")
            st.markdown("• 110% net revenue retention")
            st.metric("12-Month ARR", "$12M", delta="+100%")

        with scenario_col3:
            st.markdown("**🔴 Conservative Scenario**")
            st.markdown("• 10% monthly growth rate")
            st.markdown("• 12% monthly churn rate")
            st.markdown("• 95% net revenue retention")
            st.metric("12-Month ARR", "$8M", delta="+33%")

    def render_alerts_forecasts(self):
        """Render alerts and forecasting dashboard"""
        st.header("🚨 Alerts & Predictive Forecasts")

        # Generate alerts
        alerts = self.generate_alerts()

        # Alert summary
        col1, col2, col3 = st.columns(3)

        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in alerts if a['severity'] == 'warning']
        info_alerts = [a for a in alerts if a['severity'] == 'info']

        with col1:
            st.metric("Critical Alerts", len(critical_alerts), delta="Immediate action required")

        with col2:
            st.metric("Warning Alerts", len(warning_alerts), delta="Monitor closely")

        with col3:
            st.metric("Info Alerts", len(info_alerts), delta="Informational")

        # Display alerts
        if alerts:
            st.subheader("🚨 Active Alerts")
            for alert in alerts:
                severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}[alert['severity']]

                with st.expander(f"{severity_emoji} {alert['title']} - {alert['category']}"):
                    st.markdown(f"**Message:** {alert['message']}")
                    st.markdown(f"**Recommended Action:** {alert['action']}")
                    st.markdown(f"**Responsible:** {alert['owner']}")

                    if alert['severity'] == 'critical':
                        st.error("⚠️ Immediate action required!")

        # Predictive analytics
        st.subheader("🔮 Predictive Analytics")

        # Churn prediction
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚠️ Churn Prediction (Next 30 Days)")
            churn_prediction = self.create_churn_prediction_chart()
            st.plotly_chart(churn_prediction, use_container_width=True)

        with col2:
            st.subheader("📈 Enrollment Forecast")
            enrollment_forecast = self.create_enrollment_forecast_chart()
            st.plotly_chart(enrollment_forecast, use_container_width=True)

        # Success probability
        st.subheader("🎯 Success Probability Analysis")
        success_metrics = {
            "Reach $25K MRR in 90 days": 85,
            "Enroll 100 patients in 60 days": 90,
            "Achieve 95% customer satisfaction": 75,
            "Complete Phase XI-A trials": 80
        }

        for metric, probability in success_metrics.items():
            color = "🟢" if probability >= 80 else "🟡" if probability >= 70 else "🔴"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"{color} **{metric}**")
            with col2:
                st.metric("Probability", f"{probability}%")

    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🎛️ Monitor Controls")

        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last 12 Months"],
            index=1
        )

        # Metric filters
        st.sidebar.header("📊 Metric Filters")
        show_revenue = st.sidebar.checkbox("Revenue Metrics", value=True)
        show_customers = st.sidebar.checkbox("Customer Success", value=True)
        show_clinical = st.sidebar.checkbox("Clinical Enrollment", value=True)

        # Alert settings
        st.sidebar.header("🚨 Alert Settings")
        alert_threshold = st.sidebar.slider("Alert Threshold", 0, 100, 75)
        auto_notifications = st.sidebar.checkbox("Auto Notifications", value=True)

        # Quick stats
        st.sidebar.header("📈 Quick Stats")
        st.sidebar.metric("Current MRR", f"${self.revenue_metrics.monthly_recurring_revenue//1000}K")
        st.sidebar.metric("Patients Enrolled", self.clinical_metrics.total_patients_enrolled)
        st.sidebar.metric("Active Sites", self.clinical_metrics.total_sites_active)

        # Export options
        st.sidebar.header("📤 Export Options")
        if st.sidebar.button("📊 Export Dashboard"):
            st.sidebar.success("Dashboard exported!")

        if st.sidebar.button("📈 Generate Report"):
            st.sidebar.success("Report generated!")

    # Chart creation methods
    def create_mrr_growth_chart(self):
        """Create MRR growth chart"""
        months = pd.date_range(start='2025-01-01', end='2025-12-01', freq='M')
        base_mrr = 5000  # Starting MRR
        growth_rate = 0.15  # 15% monthly growth

        mrr_values = [base_mrr * (1 + growth_rate) ** i for i in range(len(months))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=mrr_values,
            mode='lines+markers',
            name='Monthly Recurring Revenue',
            line=dict(width=3, color='green')
        ))

        fig.update_layout(
            title="Monthly Recurring Revenue Growth",
            xaxis_title="Month",
            yaxis_title="MRR ($)",
            yaxis_tickformat='$,.0f',
            height=400
        )

        return fig

    def create_revenue_by_tier_chart(self):
        """Create revenue by tier pie chart"""
        tier_revenue = {}
        for tier in SaaSTier:
            tier_customers = [c for c in self.customers if c.tier == tier]
            tier_revenue[tier.value.title()] = sum(c.monthly_recurring_revenue for c in tier_customers)

        fig = go.Figure(data=[go.Pie(
            labels=list(tier_revenue.keys()),
            values=list(tier_revenue.values()),
            hole=0.3
        )])

        total_revenue = sum(tier_revenue.values())
        fig.update_layout(
            title="Revenue by Tier",
            annotations=[dict(text=f'${total_revenue//1000}K<br>Total MRR', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )

        return fig

    def create_customer_distribution_chart(self):
        """Create customer count by tier chart"""
        tier_counts = {}
        for tier in SaaSTier:
            tier_counts[tier.value.title()] = len([c for c in self.customers if c.tier == tier])

        fig = go.Figure(data=[
            go.Bar(x=list(tier_counts.keys()), y=list(tier_counts.values()), marker_color='lightblue')
        ])

        fig.update_layout(
            title="Customer Count by Tier",
            xaxis_title="Tier",
            yaxis_title="Customers",
            height=300
        )

        return fig

    def create_cohort_analysis_chart(self):
        """Create revenue cohort analysis"""
        # Mock cohort data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        cohorts = ['Q4 2024', 'Q1 2025', 'Q2 2025']

        # Generate mock retention data
        data = np.random.uniform(0.7, 1.0, (len(cohorts), len(months)))

        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=months,
            y=cohorts,
            colorscale='RdYlGn',
            colorbar=dict(title="Retention Rate")
        ))

        fig.update_layout(
            title="Revenue Cohort Retention Analysis",
            xaxis_title="Months Since Acquisition",
            yaxis_title="Acquisition Cohort"
        )

        return fig

    def create_health_score_distribution(self):
        """Create customer health score distribution"""
        health_scores = [c.health_score for c in self.customers]

        fig = go.Figure(data=[go.Histogram(
            x=health_scores,
            nbinsx=10,
            marker_color='lightgreen'
        )])

        fig.update_layout(
            title="Customer Health Score Distribution",
            xaxis_title="Health Score",
            yaxis_title="Number of Customers",
            height=300
        )

        return fig

    def create_usage_patterns_chart(self):
        """Create usage patterns chart"""
        usage_data = [c.usage_percentage for c in self.customers]

        fig = go.Figure(data=[go.Box(
            y=usage_data,
            name="Usage Percentage",
            marker_color='blue'
        )])

        fig.update_layout(
            title="Customer Usage Patterns",
            yaxis_title="Usage Percentage (%)",
            height=300
        )

        return fig

    def create_churn_risk_chart(self):
        """Create churn risk analysis chart"""
        risk_levels = ['Low (0-30)', 'Medium (31-70)', 'High (71-100)']
        counts = [
            len([c for c in self.customers if c.churn_risk_score <= 30]),
            len([c for c in self.customers if 30 < c.churn_risk_score <= 70]),
            len([c for c in self.customers if c.churn_risk_score > 70])
        ]
        colors = ['green', 'orange', 'red']

        fig = go.Figure(data=[
            go.Bar(x=risk_levels, y=counts, marker_color=colors)
        ])

        fig.update_layout(
            title="Churn Risk Distribution",
            xaxis_title="Risk Level",
            yaxis_title="Number of Customers",
            height=300
        )

        return fig

    def create_enrollment_progress_chart(self):
        """Create enrollment progress over time"""
        weeks = pd.date_range(start='2025-09-01', end='2025-12-31', freq='W')
        cumulative_enrollment = np.cumsum(np.random.poisson(8, len(weeks)))

        target_enrollment = np.linspace(0, self.clinical_metrics.target_enrollment, len(weeks))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weeks,
            y=cumulative_enrollment,
            mode='lines+markers',
            name='Actual Enrollment',
            line=dict(color='blue', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=weeks,
            y=target_enrollment,
            mode='lines',
            name='Target Enrollment',
            line=dict(color='green', dash='dash')
        ))

        fig.update_layout(
            title="Clinical Enrollment Progress",
            xaxis_title="Week",
            yaxis_title="Cumulative Patients Enrolled",
            height=400
        )

        return fig

    def create_site_performance_chart(self):
        """Create site performance comparison"""
        sites = [site.site_name[:15] for site in self.clinical_sites[:5]]  # Top 5 sites
        enrollment_rates = [site.enrollment_rate for site in self.clinical_sites[:5]]

        fig = go.Figure(data=[
            go.Bar(x=sites, y=enrollment_rates, marker_color='lightcoral')
        ])

        fig.update_layout(
            title="Site Enrollment Rates",
            xaxis_title="Clinical Site",
            yaxis_title="Patients/Week",
            height=300
        )

        return fig

    def create_phase_enrollment_chart(self):
        """Create enrollment by trial phase"""
        phase_data = {}
        for phase in TrialPhase:
            phase_sites = [s for s in self.clinical_sites if s.trial_phase == phase]
            phase_data[phase.value.title()] = sum(s.current_enrollment for s in phase_sites)

        fig = go.Figure(data=[go.Pie(
            labels=list(phase_data.keys()),
            values=list(phase_data.values())
        )])

        fig.update_layout(title="Enrollment by Trial Phase")

        return fig

    def create_geographic_distribution_chart(self):
        """Create geographic distribution of sites"""
        # Mock geographic data
        states = ['CA', 'NY', 'TX', 'FL', 'MA', 'IL', 'PA', 'OH']
        site_counts = [3, 2, 2, 1, 2, 1, 1, 1]

        fig = go.Figure(data=[
            go.Bar(x=states, y=site_counts, marker_color='lightsteelblue')
        ])

        fig.update_layout(
            title="Clinical Sites by State",
            xaxis_title="State",
            yaxis_title="Number of Sites",
            height=300
        )

        return fig

    def create_revenue_projections_chart(self):
        """Create revenue projections chart"""
        months = pd.date_range(start='2025-10-01', end='2026-09-01', freq='M')

        # Base case projection
        base_growth = 0.15
        base_mrr = self.revenue_metrics.monthly_recurring_revenue
        base_projection = [base_mrr * (1 + base_growth) ** i for i in range(len(months))]

        # Optimistic projection
        opt_growth = 0.25
        opt_projection = [base_mrr * (1 + opt_growth) ** i for i in range(len(months))]

        # Conservative projection
        cons_growth = 0.10
        cons_projection = [base_mrr * (1 + cons_growth) ** i for i in range(len(months))]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=months, y=base_projection, name='Base Case', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=months, y=opt_projection, name='Optimistic', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=months, y=cons_projection, name='Conservative', line=dict(color='red')))

        fig.update_layout(
            title="12-Month Revenue Projections",
            xaxis_title="Month",
            yaxis_title="Monthly Recurring Revenue ($)",
            yaxis_tickformat='$,.0f',
            height=400
        )

        return fig

    def create_growth_drivers_chart(self):
        """Create growth drivers analysis"""
        drivers = ['New Customers', 'Upsells', 'Renewals', 'Price Increases']
        contributions = [45, 25, 20, 10]  # Percentage contribution to growth

        fig = go.Figure(data=[go.Pie(
            labels=drivers,
            values=contributions,
            hole=0.3
        )])

        fig.update_layout(title="Growth Driver Contribution")

        return fig

    def create_unit_economics_chart(self):
        """Create unit economics visualization"""
        metrics = ['LTV', 'CAC', 'Gross Margin', 'Payback Period']
        values = [
            self.revenue_metrics.customer_lifetime_value // 1000,  # LTV in K
            self.revenue_metrics.customer_acquisition_cost // 1000,  # CAC in K
            85,  # Gross margin percentage
            12   # Payback period in months
        ]

        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color=['green', 'red', 'blue', 'orange'])
        ])

        fig.update_layout(
            title="Unit Economics",
            yaxis_title="Value",
            height=300
        )

        return fig

    def create_churn_prediction_chart(self):
        """Create churn prediction chart"""
        at_risk_customers = [c for c in self.customers if c.churn_risk_score >= 50]
        customer_names = [c.organization_name[:15] for c in at_risk_customers[:5]]
        churn_probabilities = [c.churn_risk_score for c in at_risk_customers[:5]]

        colors = ['red' if p >= 70 else 'orange' for p in churn_probabilities]

        fig = go.Figure(data=[
            go.Bar(x=customer_names, y=churn_probabilities, marker_color=colors)
        ])

        fig.update_layout(
            title="Churn Risk by Customer",
            xaxis_title="Customer",
            yaxis_title="Churn Risk (%)",
            height=300
        )

        return fig

    def create_enrollment_forecast_chart(self):
        """Create enrollment forecast chart"""
        weeks = pd.date_range(start=date.today(), periods=12, freq='W')
        current_enrollment = self.clinical_metrics.total_patients_enrolled
        weekly_rate = self.clinical_metrics.enrollment_rate_weekly

        forecast = [current_enrollment + (i * weekly_rate) for i in range(len(weeks))]

        fig = go.Figure(data=[
            go.Scatter(x=weeks, y=forecast, mode='lines+markers', line=dict(color='purple'))
        ])

        fig.update_layout(
            title="Enrollment Forecast (Next 12 Weeks)",
            xaxis_title="Week",
            yaxis_title="Cumulative Patients",
            height=300
        )

        return fig

    def create_top_customers_table(self):
        """Create top customers table"""
        sorted_customers = sorted(self.customers, key=lambda x: x.monthly_recurring_revenue, reverse=True)[:5]

        data = []
        for customer in sorted_customers:
            data.append({
                "Customer": customer.organization_name,
                "Tier": customer.tier.value.title(),
                "MRR": f"${customer.monthly_recurring_revenue:,}",
                "Health Score": f"{customer.health_score}/100",
                "Usage": f"{customer.usage_percentage:.0f}%",
                "Contract End": customer.contract_end_date.strftime('%m/%d/%Y'),
                "Sites": customer.clinical_site_count
            })

        return pd.DataFrame(data)

    def create_site_details_table(self):
        """Create clinical site details table"""
        data = []
        for site in self.clinical_sites:
            enrollment_progress = (site.current_enrollment / site.target_enrollment * 100) if site.target_enrollment > 0 else 0

            data.append({
                "Site": site.site_name,
                "Location": site.location,
                "PI": site.principal_investigator,
                "Status": site.status.title(),
                "Phase": site.trial_phase.value.title(),
                "Enrolled": f"{site.current_enrollment}/{site.target_enrollment}",
                "Progress": f"{enrollment_progress:.0f}%",
                "Rate": f"{site.enrollment_rate:.1f}/week",
                "Completion": f"{site.completion_rate:.0f}%"
            })

        return pd.DataFrame(data)

    # Helper methods
    def _initialize_customers(self):
        """Initialize sample customer data"""
        return [
            SaaSCustomer(
                customer_id="CUST001",
                organization_name="Mayo Clinic",
                tier=SaaSTier.ENTERPRISE,
                status=CustomerStatus.ACTIVE,
                monthly_recurring_revenue=25000,
                contract_start_date=date(2025, 9, 1),
                contract_end_date=date(2026, 9, 1),
                trial_start_date=date(2025, 8, 1),
                trial_end_date=date(2025, 8, 31),
                billing_cycle="annual",
                seats_licensed=50,
                seats_active=45,
                usage_percentage=87,
                satisfaction_score=4.8,
                health_score=92,
                churn_risk_score=15,
                clinical_site_count=3,
                total_sessions_completed=234,
                last_login_date=date(2025, 9, 20),
                support_tickets_open=1,
                success_manager="Sarah Johnson",
                deployment_phase=TrialPhase.ACTIVE,
                created_date=datetime.now(),
                last_updated=datetime.now()
            ),
            SaaSCustomer(
                customer_id="CUST002",
                organization_name="Regional Medical Center",
                tier=SaaSTier.PROFESSIONAL,
                status=CustomerStatus.ACTIVE,
                monthly_recurring_revenue=8500,
                contract_start_date=date(2025, 9, 15),
                contract_end_date=date(2026, 3, 15),
                trial_start_date=date(2025, 8, 15),
                trial_end_date=date(2025, 9, 14),
                billing_cycle="quarterly",
                seats_licensed=20,
                seats_active=16,
                usage_percentage=68,
                satisfaction_score=4.2,
                health_score=75,
                churn_risk_score=45,
                clinical_site_count=1,
                total_sessions_completed=89,
                last_login_date=date(2025, 9, 18),
                support_tickets_open=2,
                success_manager="Mike Chen",
                deployment_phase=TrialPhase.ASSIST,
                created_date=datetime.now(),
                last_updated=datetime.now()
            ),
            SaaSCustomer(
                customer_id="CUST003",
                organization_name="Community Health Network",
                tier=SaaSTier.ESSENTIAL,
                status=CustomerStatus.AT_RISK,
                monthly_recurring_revenue=2500,
                contract_start_date=date(2025, 8, 1),
                contract_end_date=date(2025, 11, 1),
                trial_start_date=date(2025, 7, 1),
                trial_end_date=date(2025, 7, 31),
                billing_cycle="monthly",
                seats_licensed=10,
                seats_active=4,
                usage_percentage=35,
                satisfaction_score=3.8,
                health_score=45,
                churn_risk_score=75,
                clinical_site_count=1,
                total_sessions_completed=23,
                last_login_date=date(2025, 9, 10),
                support_tickets_open=3,
                success_manager="Lisa Park",
                deployment_phase=TrialPhase.SHADOW,
                created_date=datetime.now(),
                last_updated=datetime.now()
            )
        ]

    def _initialize_clinical_sites(self):
        """Initialize sample clinical site data"""
        return [
            ClinicalSite(
                site_id="SITE001",
                site_name="Mayo Clinic Rochester",
                location="Rochester, MN",
                principal_investigator="Dr. David Knopman",
                site_type="academic",
                activation_date=date(2025, 9, 1),
                status="active",
                target_enrollment=30,
                current_enrollment=15,
                enrollment_rate=2.5,
                trial_phase=TrialPhase.ACTIVE,
                irb_approval_date=date(2025, 8, 15),
                equipment_installed=True,
                staff_trained=True,
                first_patient_date=date(2025, 9, 3),
                last_patient_date=date(2025, 9, 19),
                completion_rate=94,
                safety_incidents=0,
                data_quality_score=98,
                site_coordinator="Jennifer Wilson",
                created_date=datetime.now()
            ),
            ClinicalSite(
                site_id="SITE002",
                site_name="Cleveland Clinic",
                location="Cleveland, OH",
                principal_investigator="Dr. Imad Najm",
                site_type="academic",
                activation_date=date(2025, 9, 15),
                status="active",
                target_enrollment=25,
                current_enrollment=8,
                enrollment_rate=1.8,
                trial_phase=TrialPhase.ASSIST,
                irb_approval_date=date(2025, 8, 30),
                equipment_installed=True,
                staff_trained=True,
                first_patient_date=date(2025, 9, 17),
                last_patient_date=date(2025, 9, 20),
                completion_rate=89,
                safety_incidents=0,
                data_quality_score=95,
                site_coordinator="Robert Chen",
                created_date=datetime.now()
            ),
            ClinicalSite(
                site_id="SITE003",
                site_name="Stanford Medical Center",
                location="Stanford, CA",
                principal_investigator="Dr. Helen Bronte-Stewart",
                site_type="academic",
                activation_date=None,
                status="pending",
                target_enrollment=20,
                current_enrollment=0,
                enrollment_rate=0,
                trial_phase=TrialPhase.SHADOW,
                irb_approval_date=None,
                equipment_installed=False,
                staff_trained=False,
                first_patient_date=None,
                last_patient_date=None,
                completion_rate=0,
                safety_incidents=0,
                data_quality_score=0,
                site_coordinator="Amanda Rodriguez",
                created_date=datetime.now()
            )
        ]

    def _calculate_revenue_metrics(self):
        """Calculate revenue metrics"""
        total_mrr = sum(c.monthly_recurring_revenue for c in self.customers if c.status == CustomerStatus.ACTIVE)
        total_customers = len([c for c in self.customers if c.status == CustomerStatus.ACTIVE])

        return RevenueMetrics(
            monthly_recurring_revenue=total_mrr,
            annual_recurring_revenue=total_mrr * 12,
            total_customers=total_customers,
            new_customers_this_month=2,
            churned_customers_this_month=0,
            net_revenue_retention=1.15,
            gross_revenue_retention=0.92,
            customer_acquisition_cost=15000,
            customer_lifetime_value=180000,
            average_revenue_per_user=total_mrr // total_customers if total_customers > 0 else 0,
            monthly_growth_rate=18.5,
            churn_rate=5.2
        )

    def _calculate_clinical_metrics(self):
        """Calculate clinical metrics"""
        active_sites = len([s for s in self.clinical_sites if s.status == 'active'])
        total_enrolled = sum(s.current_enrollment for s in self.clinical_sites)
        target_total = sum(s.target_enrollment for s in self.clinical_sites)

        return ClinicalMetrics(
            total_sites_active=active_sites,
            total_patients_enrolled=total_enrolled,
            enrollment_rate_weekly=12.5,
            target_enrollment=target_total,
            enrollment_percentage=(total_enrolled / target_total * 100) if target_total > 0 else 0,
            sites_pending_activation=len([s for s in self.clinical_sites if s.status == 'pending']),
            completion_rate=91.5,
            dropout_rate=8.5,
            safety_incidents=0,
            protocol_deviations=2,
            data_quality_average=96.5
        )

    def generate_alerts(self):
        """Generate system alerts"""
        alerts = []

        # Revenue alerts
        for customer in self.customers:
            if customer.churn_risk_score >= 70:
                alerts.append({
                    'severity': 'critical',
                    'category': 'Customer Success',
                    'title': 'High Churn Risk',
                    'message': f"{customer.organization_name} has {customer.churn_risk_score}% churn risk",
                    'action': 'Schedule immediate customer success intervention',
                    'owner': customer.success_manager
                })

            if customer.usage_percentage < 50:
                alerts.append({
                    'severity': 'warning',
                    'category': 'Usage',
                    'title': 'Low Usage',
                    'message': f"{customer.organization_name} usage at {customer.usage_percentage:.0f}%",
                    'action': 'Provide additional training and support',
                    'owner': customer.success_manager
                })

        # Clinical alerts
        for site in self.clinical_sites:
            if site.status == 'active' and site.enrollment_rate < 1.0:
                alerts.append({
                    'severity': 'warning',
                    'category': 'Clinical',
                    'title': 'Low Enrollment Rate',
                    'message': f"{site.site_name} enrolling {site.enrollment_rate:.1f} patients/week",
                    'action': 'Review enrollment strategies with site coordinator',
                    'owner': site.site_coordinator
                })

        # Growth alerts
        if self.revenue_metrics.monthly_growth_rate < 15:
            alerts.append({
                'severity': 'warning',
                'category': 'Growth',
                'title': 'Growth Rate Below Target',
                'message': f"Monthly growth rate at {self.revenue_metrics.monthly_growth_rate:.1f}%",
                'action': 'Review sales and marketing strategies',
                'owner': 'VP Sales'
            })

        return alerts

def main():
    """Main application entry point"""
    monitor = SaaSClinicalMonitor()
    monitor.render_dashboard()

if __name__ == "__main__":
    main()