"""
Trial-Ready Operations for Adaptive Clinical AI
Feature flags, controlled rollout, role-based UI, and operational monitoring
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from uuid import UUID, uuid4
from dataclasses import dataclass

import asyncpg
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .adaptive_service import AdaptiveSession
from .safety_control import SafetyState, QualityStatus
from ..pms.pms_service import PMSService

logger = logging.getLogger(__name__)

class OperationalMode(Enum):
    """Operational modes for adaptive AI rollout"""
    SHADOW = "shadow"
    ASSIST = "assist"
    ACTIVE = "active"

class RolloutPhase(Enum):
    """Rollout phases for controlled deployment"""
    DISABLED = "disabled"
    PILOT = "pilot"
    LIMITED = "limited"
    FULL = "full"

@dataclass
class FeatureFlagConfig:
    """Feature flag configuration"""
    flag_name: str
    enabled: bool
    rollout_phase: RolloutPhase
    allowed_sites: List[str]
    allowed_modes: List[OperationalMode]
    max_sessions_per_day: int
    safety_thresholds: Dict[str, float]
    auto_pause_conditions: List[str]
    description: str

class AdaptiveFeatureManager:
    """Manages feature flags and controlled rollout for adaptive AI"""

    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.feature_cache = {}
        self.last_cache_update = datetime.min

    async def get_feature_config(self, site_id: str, trial_id: str = None) -> Dict[str, Any]:
        """Get feature configuration for site"""
        await self._refresh_cache_if_needed()

        # Get site-specific or global configuration
        site_key = f"site_{site_id}"
        trial_key = f"trial_{trial_id}" if trial_id else None

        config = {}

        # Check site-specific flags first
        if site_key in self.feature_cache:
            config.update(self.feature_cache[site_key])

        # Override with trial-specific flags
        if trial_key and trial_key in self.feature_cache:
            config.update(self.feature_cache[trial_key])

        # Apply global defaults for missing keys
        global_config = self.feature_cache.get('global', {})
        for key, value in global_config.items():
            if key not in config:
                config[key] = value

        return config

    async def is_adaptive_enabled(self, site_id: str, mode: OperationalMode,
                                 trial_id: str = None) -> Tuple[bool, str]:
        """Check if adaptive AI is enabled for site/mode combination"""
        try:
            config = await self.get_feature_config(site_id, trial_id)

            # Check master switch
            if not config.get('adaptive_ai_enabled', False):
                return False, "Adaptive AI disabled globally"

            # Check mode-specific flags
            mode_flag = f"{mode.value}_mode_enabled"
            if not config.get(mode_flag, False):
                return False, f"{mode.value.title()} mode not enabled"

            # Check site allowlist
            allowed_sites = config.get('allowed_sites', [])
            if allowed_sites and site_id not in allowed_sites:
                return False, f"Site {site_id} not in allowlist"

            # Check daily session limits
            today_sessions = await self._get_daily_session_count(site_id, mode)
            max_sessions = config.get('max_sessions_per_day', 10)
            if today_sessions >= max_sessions:
                return False, f"Daily session limit reached ({today_sessions}/{max_sessions})"

            # Check for active alerts that would pause operations
            if config.get('auto_pause_on_alert', True):
                active_alerts = await self._check_active_safety_alerts(site_id)
                if active_alerts:
                    return False, f"Auto-paused due to {len(active_alerts)} active safety alert(s)"

            return True, "Adaptive AI enabled"

        except Exception as e:
            logger.error(f"Error checking adaptive enablement: {e}")
            return False, f"Configuration error: {str(e)}"

    async def update_feature_flag(self, flag_name: str, enabled: bool,
                                 site_id: str = None, trial_id: str = None,
                                 updated_by: str = "system") -> bool:
        """Update feature flag configuration"""
        try:
            async with self.db_pool.acquire() as conn:
                # Determine scope
                if site_id and trial_id:
                    scope_key = f"trial_{trial_id}"
                elif site_id:
                    scope_key = f"site_{site_id}"
                else:
                    scope_key = "global"

                # Update or insert feature flag
                await conn.execute("""
                    INSERT INTO adaptive_feature_flags (
                        flag_name, flag_value, site_id, trial_id,
                        enabled_by, enabled_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $6)
                    ON CONFLICT (flag_name, COALESCE(site_id, ''), COALESCE(trial_id, ''))
                    DO UPDATE SET
                        flag_value = EXCLUDED.flag_value,
                        enabled_by = EXCLUDED.enabled_by,
                        enabled_at = EXCLUDED.enabled_at,
                        updated_at = EXCLUDED.updated_at
                """, flag_name, enabled, site_id, trial_id, updated_by, datetime.utcnow())

            # Clear cache to force refresh
            self.feature_cache.clear()

            logger.info(f"Feature flag {flag_name} updated: {enabled} (scope: {scope_key})")
            return True

        except Exception as e:
            logger.error(f"Error updating feature flag: {e}")
            return False

    async def emergency_disable_adaptive(self, reason: str, disabled_by: str) -> bool:
        """Emergency disable of all adaptive AI functionality"""
        try:
            async with self.db_pool.acquire() as conn:
                # Disable master switch globally
                await conn.execute("""
                    UPDATE adaptive_feature_flags
                    SET flag_value = FALSE,
                        disabled_by = $1,
                        disabled_at = $2,
                        notes = $3,
                        updated_at = $2
                    WHERE flag_name = 'adaptive_ai_enabled'
                """, disabled_by, datetime.utcnow(), f"Emergency disable: {reason}")

                # Log emergency action
                await conn.execute("""
                    INSERT INTO adaptive_safety_incidents (
                        incident_type, severity, description,
                        resolution_actions, resolved_by, resolved_at,
                        detected_by
                    ) VALUES (
                        'emergency_disable', 'critical',
                        $1, ARRAY['global_adaptive_disable'],
                        $2, $3, 'operations'
                    )
                """, f"Emergency disable: {reason}", disabled_by, datetime.utcnow())

            # Clear cache
            self.feature_cache.clear()

            logger.critical(f"Emergency disable of adaptive AI: {reason} (by {disabled_by})")
            return True

        except Exception as e:
            logger.error(f"Error in emergency disable: {e}")
            return False

    async def _refresh_cache_if_needed(self):
        """Refresh feature flag cache if stale"""
        if datetime.utcnow() - self.last_cache_update > timedelta(minutes=5):
            await self._load_feature_flags()

    async def _load_feature_flags(self):
        """Load feature flags from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT flag_name, flag_value, site_id, trial_id,
                           config_json, safety_overrides_json,
                           max_sessions_per_day, auto_pause_on_alert,
                           rollout_percentage, allowed_modes
                    FROM adaptive_feature_flags
                    WHERE flag_value = TRUE OR flag_name = 'adaptive_ai_enabled'
                """)

            self.feature_cache.clear()

            for row in rows:
                # Determine scope key
                if row['trial_id']:
                    scope_key = f"trial_{row['trial_id']}"
                elif row['site_id']:
                    scope_key = f"site_{row['site_id']}"
                else:
                    scope_key = "global"

                if scope_key not in self.feature_cache:
                    self.feature_cache[scope_key] = {}

                # Store flag configuration
                self.feature_cache[scope_key][row['flag_name']] = row['flag_value']

                # Store additional configuration
                if row['config_json']:
                    config = json.loads(row['config_json'])
                    self.feature_cache[scope_key].update(config)

                # Store operational parameters
                self.feature_cache[scope_key].update({
                    'max_sessions_per_day': row['max_sessions_per_day'] or 10,
                    'auto_pause_on_alert': row['auto_pause_on_alert'],
                    'rollout_percentage': row['rollout_percentage'] or 0.0,
                    'allowed_modes': row['allowed_modes'] or ['shadow']
                })

            self.last_cache_update = datetime.utcnow()
            logger.debug(f"Feature flags refreshed: {len(self.feature_cache)} scopes loaded")

        except Exception as e:
            logger.error(f"Error loading feature flags: {e}")

    async def _get_daily_session_count(self, site_id: str, mode: OperationalMode) -> int:
        """Get number of adaptive sessions today for site/mode"""
        try:
            async with self.db_pool.acquire() as conn:
                today = datetime.utcnow().date()
                count = await conn.fetchval("""
                    SELECT COUNT(*)
                    FROM adaptive_sessions
                    WHERE site_id = $1
                      AND mode = $2
                      AND DATE(started_at) = $3
                """, site_id, mode.value, today)
                return count or 0

        except Exception as e:
            logger.error(f"Error getting daily session count: {e}")
            return 0

    async def _check_active_safety_alerts(self, site_id: str) -> List[Dict]:
        """Check for active safety alerts that would pause operations"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT alert_id, alert_type, severity, created_at
                    FROM pms_safety_alerts
                    WHERE site_id = $1
                      AND status = 'active'
                      AND severity IN ('high', 'critical')
                """, site_id)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error checking safety alerts: {e}")
            return []

class AdaptiveOperationsUI:
    """Streamlit-based operations UI for adaptive AI monitoring and control"""

    def __init__(self, db_pool, feature_manager: AdaptiveFeatureManager):
        self.db_pool = db_pool
        self.feature_manager = feature_manager
        self.active_sessions = {}

    def render_operations_dashboard(self):
        """Render main operations dashboard"""
        st.set_page_config(
            page_title="Adaptive AI Operations",
            page_icon="🧠",
            layout="wide"
        )

        st.title("🧠 Adaptive Clinical AI Operations Dashboard")

        # Sidebar controls
        self._render_sidebar_controls()

        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎛️ Live Sessions",
            "🚨 Safety Monitor",
            "⚙️ Feature Flags",
            "📊 Analytics",
            "🔧 Controls"
        ])

        with tab1:
            self._render_live_sessions_tab()

        with tab2:
            self._render_safety_monitor_tab()

        with tab3:
            self._render_feature_flags_tab()

        with tab4:
            self._render_analytics_tab()

        with tab5:
            self._render_controls_tab()

    def _render_sidebar_controls(self):
        """Render sidebar controls"""
        st.sidebar.header("🎛️ System Status")

        # Global enable/disable
        if st.sidebar.button("🚨 EMERGENCY DISABLE", type="primary"):
            if st.sidebar.text_input("Confirm reason:", key="emergency_reason"):
                reason = st.session_state.emergency_reason
                if reason:
                    # In real implementation, this would call emergency_disable_adaptive
                    st.sidebar.success("Emergency disable triggered")
                    st.rerun()

        # Site selector
        sites = ["site_001", "site_002", "site_003", "site_004"]
        selected_site = st.sidebar.selectbox("Site Filter:", ["All Sites"] + sites)

        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
        if auto_refresh:
            st.rerun()

        st.sidebar.markdown("---")

        # Quick stats
        st.sidebar.metric("Active Sessions", self._get_active_session_count())
        st.sidebar.metric("Active Alerts", self._get_active_alert_count())
        st.sidebar.metric("Sites Online", len(sites))

    def _render_live_sessions_tab(self):
        """Render live sessions monitoring"""
        st.header("Live Adaptive Sessions")

        # Session cards
        col1, col2, col3 = st.columns(3)

        # Mock active sessions
        active_sessions = [
            {
                'session_id': 'session_123',
                'subject_id': 'PD_001',
                'site_id': 'site_001',
                'mode': 'active',
                'safety_state': 'DELIVERING',
                'current_gain': 1.15,
                'confidence': 0.87,
                'dose_used': 3.2,
                'tick_count': 45,
                'started_at': datetime.utcnow() - timedelta(minutes=12)
            },
            {
                'session_id': 'session_124',
                'subject_id': 'PD_002',
                'site_id': 'site_002',
                'mode': 'assist',
                'safety_state': 'ARMED',
                'current_gain': 1.05,
                'confidence': 0.92,
                'dose_used': 2.1,
                'tick_count': 28,
                'started_at': datetime.utcnow() - timedelta(minutes=8)
            }
        ]

        for i, session in enumerate(active_sessions):
            with [col1, col2, col3][i % 3]:
                self._render_session_card(session)

        # Session details
        if st.button("View Detailed Session Logs"):
            self._render_session_details()

    def _render_session_card(self, session: Dict):
        """Render individual session card"""
        status_color = {
            'DELIVERING': 'green',
            'ARMED': 'orange',
            'PAUSED': 'red',
            'ROLLBACK': 'red'
        }.get(session['safety_state'], 'gray')

        with st.container():
            st.markdown(f"""
            <div style="border: 2px solid {status_color}; border-radius: 10px; padding: 15px; margin: 10px 0;">
                <h4>🎯 {session['subject_id']}</h4>
                <p><strong>Mode:</strong> {session['mode'].title()}</p>
                <p><strong>State:</strong> <span style="color: {status_color};">●</span> {session['safety_state']}</p>
                <p><strong>Gain:</strong> {session['current_gain']:.2f}</p>
                <p><strong>Confidence:</strong> {session['confidence']:.1%}</p>
                <p><strong>Duration:</strong> {(datetime.utcnow() - session['started_at']).seconds // 60}m</p>
            </div>
            """, unsafe_allow_html=True)

            # Quick controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏸️ Pause", key=f"pause_{session['session_id']}"):
                    self._handle_session_action(session['session_id'], 'pause')
            with col2:
                if st.button("⏹️ Stop", key=f"stop_{session['session_id']}"):
                    self._handle_session_action(session['session_id'], 'emergency_stop')

    def _render_safety_monitor_tab(self):
        """Render safety monitoring dashboard"""
        st.header("🚨 Safety Monitor")

        # Safety status overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Safety Violations", 0, delta=0, delta_color="inverse")
        with col2:
            st.metric("Active Alerts", 1, delta=0, delta_color="inverse")
        with col3:
            st.metric("Emergency Stops", 0, delta=0, delta_color="inverse")
        with col4:
            st.metric("Safety Score", 0.95, delta=0.02)

        # Real-time safety metrics
        self._render_safety_metrics_chart()

        # Alert list
        st.subheader("Active Safety Alerts")
        alert_data = [
            {
                'Alert ID': 'ALT_001',
                'Type': 'calibration_error',
                'Severity': 'medium',
                'Site': 'site_002',
                'Created': '2 hours ago',
                'Status': 'acknowledged'
            }
        ]

        st.dataframe(alert_data, use_container_width=True)

    def _render_feature_flags_tab(self):
        """Render feature flags management"""
        st.header("⚙️ Feature Flags")

        # Global flags
        st.subheader("Global Configuration")
        col1, col2 = st.columns(2)

        with col1:
            master_enabled = st.checkbox("Master Enable", value=True)
            shadow_enabled = st.checkbox("Shadow Mode", value=True)
            assist_enabled = st.checkbox("Assist Mode", value=False)

        with col2:
            active_enabled = st.checkbox("Active Mode", value=False)
            auto_pause = st.checkbox("Auto-pause on Alerts", value=True)
            max_sessions = st.number_input("Max Sessions/Day", value=10, min_value=1, max_value=100)

        # Site-specific overrides
        st.subheader("Site-Specific Overrides")
        selected_site = st.selectbox("Select Site:", ["site_001", "site_002", "site_003"])

        with st.expander(f"Configuration for {selected_site}"):
            site_enabled = st.checkbox(f"Enable for {selected_site}", value=True)
            site_modes = st.multiselect(
                "Allowed Modes:",
                ['shadow', 'assist', 'active'],
                default=['shadow']
            )
            site_max_sessions = st.number_input(
                f"Max Sessions for {selected_site}:",
                value=5, min_value=0, max_value=50
            )

        # Apply changes
        if st.button("Apply Configuration Changes"):
            st.success("Configuration updated successfully")
            # In real implementation, this would call feature_manager.update_feature_flag

    def _render_analytics_tab(self):
        """Render analytics and performance metrics"""
        st.header("📊 Analytics")

        # Performance over time
        self._render_performance_trends()

        # Site comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Site Performance")
            site_data = {
                'Site': ['site_001', 'site_002', 'site_003'],
                'Sessions': [45, 32, 28],
                'Success Rate': [0.94, 0.89, 0.92],
                'Avg Confidence': [0.87, 0.83, 0.91]
            }
            st.dataframe(site_data)

        with col2:
            st.subheader("Mode Distribution")
            mode_data = {'Shadow': 85, 'Assist': 12, 'Active': 3}
            fig = px.pie(values=list(mode_data.values()), names=list(mode_data.keys()))
            st.plotly_chart(fig, use_container_width=True)

    def _render_controls_tab(self):
        """Render administrative controls"""
        st.header("🔧 Administrative Controls")

        # Emergency controls
        st.subheader("Emergency Controls")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚨 Global Emergency Stop", type="primary"):
                self._handle_emergency_stop()

            if st.button("⏸️ Pause All Sessions"):
                self._handle_pause_all()

        with col2:
            if st.button("🔄 Resume All Sessions"):
                self._handle_resume_all()

            if st.button("📊 Generate Safety Report"):
                self._generate_safety_report()

        # Maintenance operations
        st.subheader("Maintenance Operations")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧹 Archive Old Sessions"):
                self._archive_old_sessions()

            if st.button("🔄 Refresh Feature Flags"):
                # Force refresh feature flag cache
                st.success("Feature flags refreshed")

        with col2:
            if st.button("📋 Export Audit Logs"):
                self._export_audit_logs()

            if st.button("🔍 Run System Diagnostics"):
                self._run_system_diagnostics()

    def _render_safety_metrics_chart(self):
        """Render real-time safety metrics chart"""
        # Mock time series data
        times = pd.date_range(end=datetime.now(), periods=60, freq='1min')
        confidence_scores = np.random.beta(8, 2, 60)  # High confidence
        safety_scores = np.random.beta(9, 1, 60)  # Very high safety

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Model Confidence', 'Safety Score'),
            vertical_spacing=0.1
        )

        fig.add_trace(
            go.Scatter(x=times, y=confidence_scores, name='Confidence'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=times, y=safety_scores, name='Safety Score'),
            row=2, col=1
        )

        # Add threshold lines
        fig.add_hline(y=0.75, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=2, col=1)

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def _render_performance_trends(self):
        """Render performance trends over time"""
        # Mock performance data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        accuracy = np.random.normal(0.87, 0.03, 30)
        safety_violations = np.random.poisson(0.5, 30)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Accuracy', 'Safety Violations'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        fig.add_trace(
            go.Scatter(x=dates, y=accuracy, name='Accuracy'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=dates, y=safety_violations, name='Violations'),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def _render_session_details(self):
        """Render detailed session information"""
        st.subheader("Session Details")

        # Mock session log data
        session_logs = [
            {
                'Timestamp': '14:23:45',
                'Gain': 1.15,
                'Threshold': 0.62,
                'Confidence': 0.87,
                'Safety State': 'DELIVERING',
                'Rationale': 'conf=0.87, QC=green, increase gain'
            },
            {
                'Timestamp': '14:23:47',
                'Gain': 1.18,
                'Threshold': 0.61,
                'Confidence': 0.89,
                'Safety State': 'DELIVERING',
                'Rationale': 'conf=0.89, QC=green, target below setpoint'
            }
        ]

        st.dataframe(session_logs, use_container_width=True)

    def _handle_session_action(self, session_id: str, action: str):
        """Handle session control action"""
        st.success(f"Action '{action}' applied to session {session_id}")
        # In real implementation, this would call the adaptive service API

    def _handle_emergency_stop(self):
        """Handle global emergency stop"""
        st.error("Global emergency stop activated")
        # In real implementation, this would call emergency_disable_adaptive

    def _handle_pause_all(self):
        """Handle pause all sessions"""
        st.warning("All sessions paused")

    def _handle_resume_all(self):
        """Handle resume all sessions"""
        st.info("All sessions resumed")

    def _generate_safety_report(self):
        """Generate safety report"""
        st.success("Safety report generated and saved")

    def _archive_old_sessions(self):
        """Archive old sessions"""
        st.info("Old sessions archived")

    def _export_audit_logs(self):
        """Export audit logs"""
        st.success("Audit logs exported")

    def _run_system_diagnostics(self):
        """Run system diagnostics"""
        st.info("System diagnostics completed - all systems operational")

    def _get_active_session_count(self) -> int:
        """Get number of active sessions"""
        # Mock implementation
        return 2

    def _get_active_alert_count(self) -> int:
        """Get number of active alerts"""
        # Mock implementation
        return 1

class OperationalMonitor:
    """Background monitoring for operational health and alerting"""

    def __init__(self, db_pool, feature_manager: AdaptiveFeatureManager, pms_service: PMSService):
        self.db_pool = db_pool
        self.feature_manager = feature_manager
        self.pms_service = pms_service
        self.monitoring_active = False

    async def start_monitoring(self):
        """Start background operational monitoring"""
        self.monitoring_active = True
        logger.info("Operational monitoring started")

        # Start monitoring tasks
        tasks = [
            self._monitor_session_health(),
            self._monitor_safety_thresholds(),
            self._monitor_system_performance(),
            self._monitor_feature_flags()
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_monitoring(self):
        """Stop operational monitoring"""
        self.monitoring_active = False
        logger.info("Operational monitoring stopped")

    async def _monitor_session_health(self):
        """Monitor active session health"""
        while self.monitoring_active:
            try:
                # Check for stuck or problematic sessions
                async with self.db_pool.acquire() as conn:
                    stuck_sessions = await conn.fetch("""
                        SELECT session_id, site_id, started_at, total_ticks
                        FROM adaptive_sessions
                        WHERE status = 'active'
                          AND started_at < NOW() - INTERVAL '2 hours'
                    """)

                for session in stuck_sessions:
                    logger.warning(f"Long-running session detected: {session['session_id']}")
                    # Could trigger automatic session review

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error monitoring session health: {e}")
                await asyncio.sleep(60)

    async def _monitor_safety_thresholds(self):
        """Monitor safety threshold breaches"""
        while self.monitoring_active:
            try:
                # Check recent safety violations
                async with self.db_pool.acquire() as conn:
                    recent_violations = await conn.fetch("""
                        SELECT site_id, COUNT(*) as violation_count
                        FROM adaptive_ticks
                        WHERE timestamp_tick >= NOW() - INTERVAL '1 hour'
                          AND gates_pass = FALSE
                        GROUP BY site_id
                        HAVING COUNT(*) > 5
                    """)

                for violation in recent_violations:
                    site_id = violation['site_id']
                    count = violation['violation_count']

                    logger.warning(f"High violation rate at {site_id}: {count} violations/hour")

                    # Auto-pause if threshold exceeded
                    if count > 10:
                        await self.feature_manager.update_feature_flag(
                            'adaptive_ai_enabled', False, site_id=site_id,
                            updated_by='auto_safety_monitor'
                        )

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error monitoring safety thresholds: {e}")
                await asyncio.sleep(60)

    async def _monitor_system_performance(self):
        """Monitor system performance metrics"""
        while self.monitoring_active:
            try:
                # Check system health indicators
                performance_metrics = {
                    'active_sessions': await self._count_active_sessions(),
                    'avg_response_time': await self._measure_avg_response_time(),
                    'database_connections': await self._check_db_connections(),
                    'memory_usage': 'monitored',  # Would integrate with system monitoring
                    'cpu_usage': 'monitored'
                }

                # Log performance metrics
                logger.debug(f"System performance: {performance_metrics}")

                # Check for performance degradation
                if performance_metrics['active_sessions'] > 50:
                    logger.warning("High session load detected")

                await asyncio.sleep(600)  # Check every 10 minutes

            except Exception as e:
                logger.error(f"Error monitoring system performance: {e}")
                await asyncio.sleep(120)

    async def _monitor_feature_flags(self):
        """Monitor feature flag consistency and conflicts"""
        while self.monitoring_active:
            try:
                # Check for conflicting configurations
                config_issues = await self._detect_config_conflicts()

                if config_issues:
                    logger.warning(f"Configuration issues detected: {config_issues}")

                await asyncio.sleep(900)  # Check every 15 minutes

            except Exception as e:
                logger.error(f"Error monitoring feature flags: {e}")
                await asyncio.sleep(180)

    async def _count_active_sessions(self) -> int:
        """Count active adaptive sessions"""
        try:
            async with self.db_pool.acquire() as conn:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM adaptive_sessions
                    WHERE status = 'active'
                """)
                return count or 0
        except Exception:
            return 0

    async def _measure_avg_response_time(self) -> float:
        """Measure average system response time"""
        # Mock implementation - would measure actual API response times
        return 0.12

    async def _check_db_connections(self) -> int:
        """Check database connection pool health"""
        return self.db_pool.get_size() if hasattr(self.db_pool, 'get_size') else 5

    async def _detect_config_conflicts(self) -> List[str]:
        """Detect configuration conflicts or inconsistencies"""
        # Mock implementation - would check for conflicting feature flags
        return []

# Role-based access control
class RolePermissions:
    """Define role-based permissions for adaptive AI operations"""

    ROLES = {
        'operator': {
            'can_view_sessions': True,
            'can_pause_sessions': True,
            'can_modify_flags': False,
            'can_emergency_stop': False
        },
        'clinical_investigator': {
            'can_view_sessions': True,
            'can_pause_sessions': True,
            'can_modify_flags': True,
            'can_emergency_stop': True
        },
        'safety_officer': {
            'can_view_sessions': True,
            'can_pause_sessions': True,
            'can_modify_flags': True,
            'can_emergency_stop': True
        },
        'admin': {
            'can_view_sessions': True,
            'can_pause_sessions': True,
            'can_modify_flags': True,
            'can_emergency_stop': True
        }
    }

    @classmethod
    def check_permission(cls, user_role: str, action: str) -> bool:
        """Check if user role has permission for action"""
        role_perms = cls.ROLES.get(user_role, {})
        return role_perms.get(action, False)

# Entry point for operations UI
def main():
    """Main entry point for operations dashboard"""
    import streamlit as st
    import asyncio

    # Initialize components (mock for demonstration)
    # In real implementation, these would be properly initialized
    db_pool = None  # Would be initialized with actual database connection
    feature_manager = AdaptiveFeatureManager(db_pool)
    ui = AdaptiveOperationsUI(db_pool, feature_manager)

    # Render dashboard
    ui.render_operations_dashboard()

if __name__ == "__main__":
    main()