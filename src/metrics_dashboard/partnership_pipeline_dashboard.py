"""
Partnership Pipeline Dashboard with Real-Time LOI Tracking
EEG Neurofeedback Adaptive AI Platform

Real-time partnership pipeline management with deal progression tracking,
LOI monitoring, and weighted value calculations across four partnership tracks.

Key Features:
- Live partnership pipeline tracking with weighted valuations
- LOI progression monitoring with automated alerts
- Four-track pipeline management (MedTech, Health Systems, Tech Platforms, Telehealth)
- Deal velocity analysis and conversion rate optimization
- Partnership relationship scoring and engagement tracking
- Real-time notification system for critical milestones

Usage:
    streamlit run partnership_pipeline_dashboard.py --server.port 8503
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

class PartnershipTrack(Enum):
    """Partnership track categories"""
    MEDTECH = "medtech"
    HEALTH_SYSTEMS = "health_systems"
    TECH_PLATFORMS = "tech_platforms"
    TELEHEALTH = "telehealth"

class PartnershipStage(Enum):
    """Partnership progression stages"""
    IDENTIFIED = "identified"          # 5-15% probability
    CONTACTED = "contacted"            # 15-30% probability
    ENGAGED = "engaged"                # 30-50% probability
    PROPOSAL = "proposal"              # 50-75% probability
    NEGOTIATION = "negotiation"        # 75-90% probability
    LOI_SIGNED = "loi_signed"         # 90-95% probability
    CLOSED = "closed"                  # 95-100% probability

@dataclass
class PartnershipOpportunity:
    """Structure for partnership opportunity tracking"""
    opportunity_id: str
    partner_name: str
    partnership_track: PartnershipTrack
    tier_priority: str  # 'tier_1', 'tier_2', 'tier_3'
    current_stage: PartnershipStage
    estimated_value_min: int
    estimated_value_max: int
    probability_percentage: int
    weighted_value: int
    expected_close_date: date
    partnership_type: str  # 'joint_development', 'licensing', 'distribution', 'strategic_investment'
    key_contact_name: str
    key_contact_title: str
    relationship_strength: str  # 'cold', 'warm', 'hot'
    last_interaction_date: Optional[date]
    next_action_required: str
    next_action_due_date: date
    internal_champion: str
    external_champion: str
    competitive_alternatives: List[str]
    risk_factors: List[str]
    success_criteria: List[str]
    created_date: datetime
    last_updated: datetime

@dataclass
class LOITracker:
    """Structure for LOI tracking"""
    loi_id: str
    opportunity_id: str
    partner_name: str
    loi_status: str  # 'draft', 'internal_review', 'sent', 'under_negotiation', 'signed', 'expired'
    loi_value: int
    key_terms: List[str]
    exclusivity_period: int  # days
    due_diligence_required: bool
    legal_review_status: str
    target_signature_date: date
    actual_signature_date: Optional[date]
    expiration_date: date
    next_milestone: str
    responsible_party: str
    escalation_required: bool

@dataclass
class PartnershipMetrics:
    """Partnership pipeline metrics"""
    total_pipeline_value: int
    weighted_pipeline_value: int
    total_opportunities: int
    active_conversations: int
    lois_in_negotiation: int
    lois_signed: int
    conversion_rate_by_stage: Dict[str, float]
    average_deal_size: int
    average_sales_cycle_days: int
    pipeline_velocity: float

class PartnershipPipelineDashboard:
    """
    Real-time partnership pipeline dashboard with LOI tracking

    Provides comprehensive monitoring of partnership opportunities across
    four tracks with real-time progression and value analysis.
    """

    def __init__(self):
        self.opportunities = self._initialize_partnership_opportunities()
        self.loi_trackers = self._initialize_loi_trackers()
        self.metrics = self._calculate_pipeline_metrics()

    def render_dashboard(self):
        """Render partnership pipeline dashboard"""
        st.title("🤝 Partnership Pipeline Dashboard")
        st.markdown("Real-time tracking of partnership opportunities and LOI progression")

        # Executive summary
        self.render_executive_summary()

        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Pipeline Overview", "📋 Opportunity Tracking", "📄 LOI Monitoring", "📊 Analytics", "🚨 Alerts"
        ])

        with tab1:
            self.render_pipeline_overview()

        with tab2:
            self.render_opportunity_tracking()

        with tab3:
            self.render_loi_monitoring()

        with tab4:
            self.render_pipeline_analytics()

        with tab5:
            self.render_alerts_and_actions()

        # Sidebar controls
        self.render_sidebar()

    def render_executive_summary(self):
        """Render executive summary with key pipeline metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Pipeline",
                f"${self.metrics.total_pipeline_value//1000000}M",
                delta=f"{self.metrics.total_opportunities} opportunities"
            )

        with col2:
            st.metric(
                "Weighted Value",
                f"${self.metrics.weighted_pipeline_value//1000000}M",
                delta=f"{(self.metrics.weighted_pipeline_value/self.metrics.total_pipeline_value*100):.0f}% probability-adjusted"
            )

        with col3:
            st.metric(
                "Active Conversations",
                f"{self.metrics.active_conversations}",
                delta=f"{len([o for o in self.opportunities if o.tier_priority == 'tier_1'])} Tier 1"
            )

        with col4:
            st.metric(
                "LOIs in Progress",
                f"{self.metrics.lois_in_negotiation}",
                delta=f"{self.metrics.lois_signed} signed"
            )

        with col5:
            avg_deal_size = self.metrics.average_deal_size // 1000000
            st.metric(
                "Avg Deal Size",
                f"${avg_deal_size}M",
                delta=f"{self.metrics.average_sales_cycle_days} day cycle"
            )

        # Pipeline health indicators
        st.markdown("---")
        self.render_pipeline_health_indicators()

    def render_pipeline_health_indicators(self):
        """Render pipeline health traffic light indicators"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tier1_engaged = len([o for o in self.opportunities
                               if o.tier_priority == 'tier_1' and o.current_stage.value in ['engaged', 'proposal', 'negotiation']])
            status = "🟢" if tier1_engaged >= 3 else "🟡" if tier1_engaged >= 2 else "🔴"
            st.markdown(f"**Tier 1 Engagement** {status}")
            st.caption(f"{tier1_engaged}/4 Tier 1 partners actively engaged")

        with col2:
            medtech_progress = len([o for o in self.opportunities
                                  if o.partnership_track == PartnershipTrack.MEDTECH and o.current_stage.value in ['proposal', 'negotiation']])
            status = "🟢" if medtech_progress >= 1 else "🟡"
            st.markdown(f"**MedTech Progress** {status}")
            st.caption(f"{medtech_progress} MedTech partnerships in advanced stages")

        with col3:
            loi_health = len([l for l in self.loi_trackers if l.loi_status in ['under_negotiation', 'signed']])
            status = "🟢" if loi_health >= 2 else "🟡" if loi_health >= 1 else "🔴"
            st.markdown(f"**LOI Health** {status}")
            st.caption(f"{loi_health} LOIs active or signed")

        with col4:
            pipeline_velocity = self.metrics.pipeline_velocity
            status = "🟢" if pipeline_velocity >= 15 else "🟡" if pipeline_velocity >= 10 else "🔴"
            st.markdown(f"**Pipeline Velocity** {status}")
            st.caption(f"{pipeline_velocity:.1f}% monthly progression rate")

    def render_pipeline_overview(self):
        """Render pipeline overview with funnel and track analysis"""
        st.header("🎯 Partnership Pipeline Overview")

        # Pipeline funnel
        st.subheader("📊 Partnership Funnel Analysis")
        funnel_chart = self.create_partnership_funnel()
        st.plotly_chart(funnel_chart, use_container_width=True)

        # Pipeline by track
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💼 Pipeline Value by Track")
            track_value_chart = self.create_pipeline_by_track_chart()
            st.plotly_chart(track_value_chart, use_container_width=True)

        with col2:
            st.subheader("🎯 Opportunity Count by Stage")
            stage_count_chart = self.create_stage_count_chart()
            st.plotly_chart(stage_count_chart, use_container_width=True)

        # Weighted value progression
        st.subheader("📈 Weighted Pipeline Value Over Time")
        value_progression = self.create_weighted_value_progression()
        st.plotly_chart(value_progression, use_container_width=True)

        # Top opportunities summary
        st.subheader("🌟 Top 10 Partnership Opportunities")
        top_opportunities = self.create_top_opportunities_table()
        st.dataframe(top_opportunities, use_container_width=True)

    def render_opportunity_tracking(self):
        """Render detailed opportunity tracking"""
        st.header("📋 Partnership Opportunity Tracking")

        # Track filter
        track_filter = st.selectbox(
            "Filter by Partnership Track",
            ["All Tracks"] + [track.value.replace('_', ' ').title() for track in PartnershipTrack],
            key="track_filter"
        )

        # Filter opportunities
        filtered_opportunities = self.opportunities
        if track_filter != "All Tracks":
            track_enum = PartnershipTrack(track_filter.lower().replace(' ', '_'))
            filtered_opportunities = [o for o in self.opportunities if o.partnership_track == track_enum]

        # Opportunity cards
        for opportunity in filtered_opportunities:
            with st.expander(f"🏢 {opportunity.partner_name} - ${opportunity.estimated_value_max//1000000}M - {opportunity.current_stage.value.title()}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    **Partnership Details:**
                    - **Track:** {opportunity.partnership_track.value.replace('_', ' ').title()}
                    - **Type:** {opportunity.partnership_type.replace('_', ' ').title()}
                    - **Priority:** {opportunity.tier_priority.replace('_', ' ').title()}
                    - **Estimated Value:** ${opportunity.estimated_value_min//1000000}-${opportunity.estimated_value_max//1000000}M
                    - **Probability:** {opportunity.probability_percentage}%
                    - **Weighted Value:** ${opportunity.weighted_value//1000000}M
                    """)

                with col2:
                    st.markdown(f"""
                    **Relationship & Progress:**
                    - **Key Contact:** {opportunity.key_contact_name} ({opportunity.key_contact_title})
                    - **Relationship:** {opportunity.relationship_strength.title()}
                    - **Last Interaction:** {opportunity.last_interaction_date.strftime('%m/%d/%Y') if opportunity.last_interaction_date else 'None'}
                    - **Internal Champion:** {opportunity.internal_champion}
                    - **External Champion:** {opportunity.external_champion}
                    - **Expected Close:** {opportunity.expected_close_date.strftime('%m/%d/%Y')}
                    """)

                # Progress visualization
                stage_progress = self.create_opportunity_progress_chart(opportunity)
                st.plotly_chart(stage_progress, use_container_width=True)

                # Next actions
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🎯 Next Action:** {opportunity.next_action_required}")
                    st.markdown(f"**📅 Due Date:** {opportunity.next_action_due_date.strftime('%m/%d/%Y')}")

                with col2:
                    if opportunity.risk_factors:
                        st.markdown("**⚠️ Risk Factors:**")
                        for risk in opportunity.risk_factors:
                            st.markdown(f"• {risk}")

                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📝 Log Interaction", key=f"log_{opportunity.opportunity_id}"):
                        st.success("Interaction logged!")

                with col2:
                    if st.button(f"📈 Advance Stage", key=f"advance_{opportunity.opportunity_id}"):
                        self.advance_opportunity_stage(opportunity.opportunity_id)

                with col3:
                    if st.button(f"📄 Generate LOI", key=f"loi_{opportunity.opportunity_id}"):
                        self.generate_loi(opportunity.opportunity_id)

    def render_loi_monitoring(self):
        """Render LOI monitoring dashboard"""
        st.header("📄 LOI Monitoring & Tracking")

        if not self.loi_trackers:
            st.info("🔵 No LOIs currently in progress. LOIs will appear here when partnerships reach proposal stage.")
            return

        # LOI summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_loi_value = sum(loi.loi_value for loi in self.loi_trackers)
            st.metric("Total LOI Value", f"${total_loi_value//1000000}M")

        with col2:
            active_lois = len([loi for loi in self.loi_trackers if loi.loi_status in ['under_negotiation', 'sent']])
            st.metric("Active LOIs", active_lois)

        with col3:
            signed_lois = len([loi for loi in self.loi_trackers if loi.loi_status == 'signed'])
            st.metric("Signed LOIs", signed_lois)

        with col4:
            avg_negotiation_time = 45  # Mock average
            st.metric("Avg Negotiation Time", f"{avg_negotiation_time} days")

        # LOI status timeline
        st.subheader("📅 LOI Status Timeline")
        loi_timeline = self.create_loi_timeline_chart()
        st.plotly_chart(loi_timeline, use_container_width=True)

        # Individual LOI tracking
        st.subheader("📋 Individual LOI Status")
        for loi in self.loi_trackers:
            with st.expander(f"📄 {loi.partner_name} LOI - ${loi.loi_value//1000000}M - {loi.loi_status.replace('_', ' ').title()}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    **LOI Details:**
                    - **Status:** {loi.loi_status.replace('_', ' ').title()}
                    - **Value:** ${loi.loi_value//1000000}M
                    - **Target Signature:** {loi.target_signature_date.strftime('%m/%d/%Y')}
                    - **Expiration:** {loi.expiration_date.strftime('%m/%d/%Y')}
                    - **Exclusivity Period:** {loi.exclusivity_period} days
                    """)

                with col2:
                    st.markdown(f"""
                    **Progress & Next Steps:**
                    - **Legal Review:** {loi.legal_review_status.title()}
                    - **Due Diligence Required:** {'Yes' if loi.due_diligence_required else 'No'}
                    - **Next Milestone:** {loi.next_milestone}
                    - **Responsible Party:** {loi.responsible_party}
                    - **Escalation Required:** {'Yes' if loi.escalation_required else 'No'}
                    """)

                # Key terms
                st.markdown("**📋 Key Terms:**")
                for term in loi.key_terms:
                    st.markdown(f"• {term}")

                # Progress gauge
                status_to_progress = {
                    'draft': 20, 'internal_review': 40, 'sent': 60,
                    'under_negotiation': 80, 'signed': 100, 'expired': 0
                }
                progress = status_to_progress.get(loi.loi_status, 0)

                progress_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = progress,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "LOI Progress"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                progress_gauge.update_layout(height=250)
                st.plotly_chart(progress_gauge, use_container_width=True)

    def render_pipeline_analytics(self):
        """Render pipeline analytics and insights"""
        st.header("📊 Pipeline Analytics & Insights")

        # Conversion rate analysis
        st.subheader("📈 Conversion Rate Analysis")
        conversion_chart = self.create_conversion_rate_chart()
        st.plotly_chart(conversion_chart, use_container_width=True)

        # Deal velocity analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚡ Deal Velocity by Track")
            velocity_chart = self.create_deal_velocity_chart()
            st.plotly_chart(velocity_chart, use_container_width=True)

        with col2:
            st.subheader("💰 Average Deal Size Trends")
            deal_size_chart = self.create_deal_size_trends_chart()
            st.plotly_chart(deal_size_chart, use_container_width=True)

        # Relationship strength analysis
        st.subheader("🤝 Relationship Strength Distribution")
        relationship_chart = self.create_relationship_strength_chart()
        st.plotly_chart(relationship_chart, use_container_width=True)

        # Win/loss analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Win Rate by Track")
            win_rate_data = {
                "Track": ["MedTech", "Health Systems", "Tech Platforms", "Telehealth"],
                "Win Rate": [35, 45, 30, 40],
                "Opportunities": [4, 6, 3, 2]
            }
            win_rate_df = pd.DataFrame(win_rate_data)
            st.dataframe(win_rate_df, use_container_width=True)

        with col2:
            st.subheader("📊 Pipeline Health Score")
            health_score = self.calculate_pipeline_health_score()
            st.metric("Overall Health Score", f"{health_score}/100", delta="Above target (75)")

            health_factors = {
                "Pipeline Value": 85,
                "Stage Progression": 78,
                "Relationship Quality": 82,
                "Conversion Rates": 71
            }

            for factor, score in health_factors.items():
                color = "🟢" if score >= 80 else "🟡" if score >= 70 else "🔴"
                st.markdown(f"{color} **{factor}:** {score}/100")

    def render_alerts_and_actions(self):
        """Render alerts and required actions"""
        st.header("🚨 Alerts & Required Actions")

        # Generate alerts
        alerts = self.generate_alerts()

        if not alerts:
            st.success("🟢 No critical alerts - pipeline is healthy!")
            return

        # Alert summary
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in alerts if a['severity'] == 'warning']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Critical Alerts", len(critical_alerts), delta="Immediate action required")

        with col2:
            st.metric("Warning Alerts", len(warning_alerts), delta="Action needed soon")

        with col3:
            overdue_actions = len([a for a in alerts if 'overdue' in a['message'].lower()])
            st.metric("Overdue Actions", overdue_actions, delta="Past due date")

        # Display alerts
        for alert in alerts:
            severity_color = {"critical": "🔴", "warning": "🟡", "info": "🔵"}[alert['severity']]

            with st.expander(f"{severity_color} {alert['title']} - {alert['partner']}"):
                st.markdown(f"**Message:** {alert['message']}")
                st.markdown(f"**Due Date:** {alert['due_date']}")
                st.markdown(f"**Responsible:** {alert['responsible_party']}")

                if alert['severity'] == 'critical':
                    st.error(f"⚠️ {alert['recommended_action']}")
                else:
                    st.info(f"💡 {alert['recommended_action']}")

                if st.button(f"Mark as Addressed", key=f"resolve_{alert['alert_id']}"):
                    st.success("Alert marked as addressed!")

        # Action items summary
        st.subheader("📋 Action Items Summary")
        action_items = self.create_action_items_table()
        st.dataframe(action_items, use_container_width=True)

    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🎛️ Pipeline Controls")

        # Time horizon selector
        time_horizon = st.sidebar.selectbox(
            "Time Horizon",
            ["This Month", "Next 3 Months", "Next 6 Months", "Full Year"],
            index=1
        )

        # Track filters
        st.sidebar.header("🎯 Track Filters")
        show_medtech = st.sidebar.checkbox("MedTech Giants", value=True)
        show_health_systems = st.sidebar.checkbox("Health Systems", value=True)
        show_tech_platforms = st.sidebar.checkbox("Tech Platforms", value=True)
        show_telehealth = st.sidebar.checkbox("Telehealth", value=True)

        # Stage filters
        st.sidebar.header("📊 Stage Filters")
        min_stage = st.sidebar.selectbox(
            "Minimum Stage",
            [stage.value.replace('_', ' ').title() for stage in PartnershipStage],
            index=0
        )

        # Quick actions
        st.sidebar.header("⚡ Quick Actions")
        if st.sidebar.button("📧 Send Pipeline Report"):
            st.sidebar.success("Pipeline report sent!")

        if st.sidebar.button("📅 Schedule Follow-ups"):
            st.sidebar.success("Follow-ups scheduled!")

        if st.sidebar.button("📊 Export Data"):
            st.sidebar.success("Data exported!")

        # Pipeline health summary
        st.sidebar.header("🏥 Pipeline Health")
        health_score = self.calculate_pipeline_health_score()
        if health_score >= 80:
            st.sidebar.success(f"🟢 Healthy ({health_score}/100)")
        elif health_score >= 70:
            st.sidebar.warning(f"🟡 Attention Needed ({health_score}/100)")
        else:
            st.sidebar.error(f"🔴 Critical ({health_score}/100)")

    # Chart creation methods
    def create_partnership_funnel(self):
        """Create partnership funnel chart"""
        stage_counts = {}
        for stage in PartnershipStage:
            stage_counts[stage.value] = len([o for o in self.opportunities if o.current_stage == stage])

        stages = list(stage_counts.keys())
        counts = list(stage_counts.values())

        fig = go.Figure(go.Funnel(
            y = [stage.replace('_', ' ').title() for stage in stages],
            x = counts,
            textinfo = "value+percent initial",
            marker_color = ["deepskyblue", "lightsalmon", "lightgreen", "gold", "orange", "red", "purple"]
        ))

        fig.update_layout(
            title="Partnership Pipeline Funnel",
            height=500
        )

        return fig

    def create_pipeline_by_track_chart(self):
        """Create pipeline value by track chart"""
        track_values = {}
        for track in PartnershipTrack:
            track_opportunities = [o for o in self.opportunities if o.partnership_track == track]
            track_values[track.value] = sum(o.estimated_value_max for o in track_opportunities) // 1000000

        tracks = [track.replace('_', ' ').title() for track in track_values.keys()]
        values = list(track_values.values())

        fig = go.Figure(data=[go.Pie(labels=tracks, values=values, hole=0.3)])

        fig.update_layout(
            title="Pipeline Value by Track ($M)",
            annotations=[dict(text=f'${sum(values)}M<br>Total', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )

        return fig

    def create_stage_count_chart(self):
        """Create opportunity count by stage chart"""
        stage_counts = {}
        for stage in PartnershipStage:
            stage_counts[stage.value.replace('_', ' ').title()] = len([o for o in self.opportunities if o.current_stage == stage])

        fig = go.Figure(data=[
            go.Bar(x=list(stage_counts.keys()), y=list(stage_counts.values()), marker_color='lightblue')
        ])

        fig.update_layout(
            title="Opportunities by Stage",
            xaxis_title="Stage",
            yaxis_title="Count",
            height=400
        )

        return fig

    def create_weighted_value_progression(self):
        """Create weighted pipeline value progression chart"""
        # Mock historical data
        months = pd.date_range(start='2025-01-01', end='2025-12-01', freq='M')
        base_value = 200  # $200M starting value
        progression = [base_value + (i * 50) + np.random.normal(0, 20) for i in range(len(months))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=progression,
            mode='lines+markers',
            name='Weighted Pipeline Value',
            line=dict(width=3, color='blue')
        ))

        fig.update_layout(
            title="Weighted Pipeline Value Progression",
            xaxis_title="Month",
            yaxis_title="Value ($M)",
            height=400
        )

        return fig

    def create_opportunity_progress_chart(self, opportunity):
        """Create individual opportunity progress chart"""
        stages = [stage.value.replace('_', ' ').title() for stage in PartnershipStage]
        current_stage_index = list(PartnershipStage).index(opportunity.current_stage)

        progress = [100 if i <= current_stage_index else 0 for i in range(len(stages))]
        colors = ['green' if p == 100 else 'lightgray' for p in progress]

        fig = go.Figure(data=[
            go.Bar(x=stages, y=[1]*len(stages), marker_color=colors, showlegend=False)
        ])

        fig.update_layout(
            title=f"Progress: {opportunity.partner_name}",
            xaxis_title="Stage",
            yaxis=dict(showticklabels=False),
            height=200
        )

        return fig

    def create_loi_timeline_chart(self):
        """Create LOI timeline chart"""
        if not self.loi_trackers:
            return go.Figure()

        fig = go.Figure()

        for loi in self.loi_trackers:
            color = {
                'draft': 'gray', 'internal_review': 'orange', 'sent': 'blue',
                'under_negotiation': 'purple', 'signed': 'green', 'expired': 'red'
            }[loi.loi_status]

            fig.add_trace(go.Scatter(
                x=[loi.target_signature_date],
                y=[loi.partner_name],
                mode='markers+text',
                marker=dict(size=12, color=color),
                text=f"${loi.loi_value//1000000}M",
                textposition="middle right",
                name=loi.loi_status.replace('_', ' ').title(),
                showlegend=False
            ))

        fig.update_layout(
            title="LOI Timeline",
            xaxis_title="Target Signature Date",
            yaxis_title="Partner",
            height=400
        )

        return fig

    def create_conversion_rate_chart(self):
        """Create conversion rate analysis chart"""
        stages = ["Identified→Contacted", "Contacted→Engaged", "Engaged→Proposal", "Proposal→Negotiation", "Negotiation→Closed"]
        rates = [75, 60, 40, 65, 80]  # Mock conversion rates
        colors = ['green' if rate >= 70 else 'orange' if rate >= 50 else 'red' for rate in rates]

        fig = go.Figure(data=[
            go.Bar(x=stages, y=rates, marker_color=colors, text=rates, textposition='auto')
        ])

        fig.update_layout(
            title="Stage Conversion Rates",
            xaxis_title="Stage Transition",
            yaxis_title="Conversion Rate (%)",
            height=400
        )

        return fig

    def create_deal_velocity_chart(self):
        """Create deal velocity by track chart"""
        tracks = ["MedTech", "Health Systems", "Tech Platforms", "Telehealth"]
        velocities = [12, 18, 15, 20]  # Mock velocities in %/month

        fig = go.Figure(data=[
            go.Bar(x=tracks, y=velocities, marker_color='lightgreen')
        ])

        fig.update_layout(
            title="Deal Velocity by Track",
            xaxis_title="Partnership Track",
            yaxis_title="Velocity (%/month)",
            height=300
        )

        return fig

    def create_deal_size_trends_chart(self):
        """Create deal size trends chart"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        avg_deal_sizes = [45, 52, 48, 58, 62, 67]  # Mock data in $M

        fig = go.Figure(data=[
            go.Scatter(x=months, y=avg_deal_sizes, mode='lines+markers', line=dict(color='blue'))
        ])

        fig.update_layout(
            title="Average Deal Size Trends",
            xaxis_title="Month",
            yaxis_title="Avg Deal Size ($M)",
            height=300
        )

        return fig

    def create_relationship_strength_chart(self):
        """Create relationship strength distribution chart"""
        strengths = ['Cold', 'Warm', 'Hot']
        counts = [
            len([o for o in self.opportunities if o.relationship_strength == 'cold']),
            len([o for o in self.opportunities if o.relationship_strength == 'warm']),
            len([o for o in self.opportunities if o.relationship_strength == 'hot'])
        ]

        fig = go.Figure(data=[go.Pie(labels=strengths, values=counts)])
        fig.update_layout(title="Relationship Strength Distribution")

        return fig

    def create_top_opportunities_table(self):
        """Create top opportunities summary table"""
        # Sort by weighted value
        sorted_opportunities = sorted(self.opportunities, key=lambda x: x.weighted_value, reverse=True)[:10]

        data = []
        for opp in sorted_opportunities:
            data.append({
                "Partner": opp.partner_name,
                "Track": opp.partnership_track.value.replace('_', ' ').title(),
                "Value": f"${opp.estimated_value_max//1000000}M",
                "Weighted": f"${opp.weighted_value//1000000}M",
                "Stage": opp.current_stage.value.replace('_', ' ').title(),
                "Probability": f"{opp.probability_percentage}%",
                "Expected Close": opp.expected_close_date.strftime('%m/%d/%Y'),
                "Priority": opp.tier_priority.replace('_', ' ').title()
            })

        return pd.DataFrame(data)

    def create_action_items_table(self):
        """Create action items summary table"""
        action_items = []

        for opp in self.opportunities:
            if opp.next_action_due_date <= date.today() + timedelta(days=7):
                urgency = "🔴 Overdue" if opp.next_action_due_date < date.today() else "🟡 Due Soon"
                action_items.append({
                    "Urgency": urgency,
                    "Partner": opp.partner_name,
                    "Action": opp.next_action_required,
                    "Due Date": opp.next_action_due_date.strftime('%m/%d/%Y'),
                    "Owner": opp.internal_champion,
                    "Stage": opp.current_stage.value.replace('_', ' ').title()
                })

        return pd.DataFrame(action_items) if action_items else pd.DataFrame({"Message": ["No urgent action items"]})

    # Helper methods
    def _initialize_partnership_opportunities(self):
        """Initialize partnership opportunities data"""
        return [
            PartnershipOpportunity(
                opportunity_id="OP001",
                partner_name="Medtronic",
                partnership_track=PartnershipTrack.MEDTECH,
                tier_priority="tier_1",
                current_stage=PartnershipStage.ENGAGED,
                estimated_value_min=100000000,
                estimated_value_max=500000000,
                probability_percentage=40,
                weighted_value=200000000,
                expected_close_date=date(2025, 12, 15),
                partnership_type="joint_development",
                key_contact_name="Brett Wall",
                key_contact_title="President, Neuroscience Portfolio",
                relationship_strength="warm",
                last_interaction_date=date(2025, 9, 15),
                next_action_required="Technology demonstration",
                next_action_due_date=date(2025, 10, 5),
                internal_champion="VP Strategic Partnerships",
                external_champion="Brett Wall",
                competitive_alternatives=["Boston Scientific", "Abbott"],
                risk_factors=["Long sales cycle", "Complex integration"],
                success_criteria=["Joint development agreement", "Regulatory pathway alignment"],
                created_date=datetime.now(),
                last_updated=datetime.now()
            ),
            PartnershipOpportunity(
                opportunity_id="OP002",
                partner_name="Mayo Clinic",
                partnership_track=PartnershipTrack.HEALTH_SYSTEMS,
                tier_priority="tier_1",
                current_stage=PartnershipStage.PROPOSAL,
                estimated_value_min=5000000,
                estimated_value_max=25000000,
                probability_percentage=65,
                weighted_value=16250000,
                expected_close_date=date(2025, 11, 30),
                partnership_type="clinical_validation",
                key_contact_name="Dr. David Knopman",
                key_contact_title="Neurologist",
                relationship_strength="warm",
                last_interaction_date=date(2025, 9, 18),
                next_action_required="Pilot program proposal review",
                next_action_due_date=date(2025, 9, 30),
                internal_champion="Clinical Director",
                external_champion="Dr. David Knopman",
                competitive_alternatives=["Standard neurofeedback", "Manual therapy"],
                risk_factors=["IRB approval timing", "Clinical resource availability"],
                success_criteria=["Pilot agreement signed", "Clinical validation study"],
                created_date=datetime.now(),
                last_updated=datetime.now()
            ),
            PartnershipOpportunity(
                opportunity_id="OP003",
                partner_name="Google Health",
                partnership_track=PartnershipTrack.TECH_PLATFORMS,
                tier_priority="tier_1",
                current_stage=PartnershipStage.CONTACTED,
                estimated_value_min=25000000,
                estimated_value_max=100000000,
                probability_percentage=25,
                weighted_value=31250000,
                expected_close_date=date(2026, 3, 15),
                partnership_type="technology_integration",
                key_contact_name="Karen DeSalvo",
                key_contact_title="Chief Health Officer",
                relationship_strength="cold",
                last_interaction_date=None,
                next_action_required="Initial partnership call",
                next_action_due_date=date(2025, 10, 10),
                internal_champion="CTO",
                external_champion="TBD",
                competitive_alternatives=["Microsoft Healthcare", "Amazon Healthcare"],
                risk_factors=["Platform priorities", "Integration complexity"],
                success_criteria=["Technology integration agreement", "Joint go-to-market"],
                created_date=datetime.now(),
                last_updated=datetime.now()
            ),
            # Additional opportunities would be added here...
        ]

    def _initialize_loi_trackers(self):
        """Initialize LOI tracking data"""
        return [
            LOITracker(
                loi_id="LOI001",
                opportunity_id="OP002",
                partner_name="Mayo Clinic",
                loi_status="under_negotiation",
                loi_value=15000000,
                key_terms=[
                    "12-month pilot program",
                    "Shared clinical data rights",
                    "Joint publication agreement",
                    "Option for 5-year extension"
                ],
                exclusivity_period=90,
                due_diligence_required=True,
                legal_review_status="in_progress",
                target_signature_date=date(2025, 10, 15),
                actual_signature_date=None,
                expiration_date=date(2025, 11, 15),
                next_milestone="Legal review completion",
                responsible_party="Legal Counsel",
                escalation_required=False
            )
        ]

    def _calculate_pipeline_metrics(self):
        """Calculate pipeline metrics"""
        total_value = sum(opp.estimated_value_max for opp in self.opportunities)
        weighted_value = sum(opp.weighted_value for opp in self.opportunities)
        active_conversations = len([opp for opp in self.opportunities
                                  if opp.current_stage.value in ['engaged', 'proposal', 'negotiation']])

        return PartnershipMetrics(
            total_pipeline_value=total_value,
            weighted_pipeline_value=weighted_value,
            total_opportunities=len(self.opportunities),
            active_conversations=active_conversations,
            lois_in_negotiation=len([loi for loi in self.loi_trackers if loi.loi_status == 'under_negotiation']),
            lois_signed=len([loi for loi in self.loi_trackers if loi.loi_status == 'signed']),
            conversion_rate_by_stage={},
            average_deal_size=total_value // len(self.opportunities) if self.opportunities else 0,
            average_sales_cycle_days=365,
            pipeline_velocity=12.5
        )

    def calculate_pipeline_health_score(self):
        """Calculate overall pipeline health score"""
        # Mock calculation based on various factors
        base_score = 75

        # Adjust for tier 1 engagement
        tier1_engaged = len([o for o in self.opportunities
                           if o.tier_priority == 'tier_1' and o.current_stage.value in ['engaged', 'proposal', 'negotiation']])
        base_score += min(15, tier1_engaged * 5)

        # Adjust for LOI activity
        active_lois = len([loi for loi in self.loi_trackers if loi.loi_status in ['under_negotiation', 'signed']])
        base_score += min(10, active_lois * 5)

        return min(100, base_score)

    def generate_alerts(self):
        """Generate alerts for critical actions"""
        alerts = []

        for opp in self.opportunities:
            # Overdue actions
            if opp.next_action_due_date < date.today():
                alerts.append({
                    'alert_id': f"ALERT_{opp.opportunity_id}_OVERDUE",
                    'severity': 'critical',
                    'title': 'Overdue Action',
                    'partner': opp.partner_name,
                    'message': f"Action '{opp.next_action_required}' is overdue by {(date.today() - opp.next_action_due_date).days} days",
                    'due_date': opp.next_action_due_date.strftime('%m/%d/%Y'),
                    'responsible_party': opp.internal_champion,
                    'recommended_action': 'Contact partner immediately to reschedule and maintain relationship'
                })

            # Stalled opportunities
            if opp.last_interaction_date and (date.today() - opp.last_interaction_date).days > 30:
                alerts.append({
                    'alert_id': f"ALERT_{opp.opportunity_id}_STALLED",
                    'severity': 'warning',
                    'title': 'Stalled Opportunity',
                    'partner': opp.partner_name,
                    'message': f"No interaction for {(date.today() - opp.last_interaction_date).days} days",
                    'due_date': date.today().strftime('%m/%d/%Y'),
                    'responsible_party': opp.internal_champion,
                    'recommended_action': 'Schedule check-in call to re-engage and assess continued interest'
                })

        # LOI expiration alerts
        for loi in self.loi_trackers:
            days_to_expiration = (loi.expiration_date - date.today()).days
            if days_to_expiration <= 7 and loi.loi_status != 'signed':
                alerts.append({
                    'alert_id': f"ALERT_{loi.loi_id}_EXPIRING",
                    'severity': 'critical',
                    'title': 'LOI Expiring Soon',
                    'partner': loi.partner_name,
                    'message': f"LOI expires in {days_to_expiration} days",
                    'due_date': loi.expiration_date.strftime('%m/%d/%Y'),
                    'responsible_party': loi.responsible_party,
                    'recommended_action': 'Expedite negotiations or request extension'
                })

        return alerts

    def advance_opportunity_stage(self, opportunity_id):
        """Advance opportunity to next stage"""
        st.success(f"Opportunity {opportunity_id} advanced to next stage!")

    def generate_loi(self, opportunity_id):
        """Generate LOI for opportunity"""
        st.success(f"LOI generated for opportunity {opportunity_id}!")

def main():
    """Main application entry point"""
    dashboard = PartnershipPipelineDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()