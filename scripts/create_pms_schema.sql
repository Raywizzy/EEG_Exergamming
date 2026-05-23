-- Phase X-C: Post-Market Surveillance Database Schema
-- Global safety monitoring, model drift detection, and real-world performance tracking
--
-- Tables designed for high-volume ingestion and fast analytical queries
-- Optimized for FDA/EMA periodic safety update reports (PSUR/PMSR)

-- ================================================================
-- Core PMS Events Table - Central logging of all PMS activities
-- ================================================================

CREATE TABLE IF NOT EXISTS pms_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id VARCHAR(50) NOT NULL,
    site_id VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    device_family VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- 'prediction', 'ground_truth', 'adverse_event', 'device_issue', 'protocol_deviation'
    event_timestamp TIMESTAMPTZ NOT NULL,
    session_id VARCHAR(100),
    trial_id VARCHAR(50),
    payload_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Computed columns for fast filtering
    event_date DATE GENERATED ALWAYS AS (DATE(event_timestamp)) STORED,
    event_hour INTEGER GENERATED ALWAYS AS (EXTRACT(HOUR FROM event_timestamp)) STORED
);

-- High-performance indexes for PMS events
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_events_date_type ON pms_events (event_date, event_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_events_site_model ON pms_events (site_id, model_version, event_timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_events_subject_session ON pms_events (subject_id, session_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_events_device_family ON pms_events (device_family, event_timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_events_payload_gin ON pms_events USING GIN (payload_json);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_events_timestamp ON pms_events (event_timestamp DESC);

-- ================================================================
-- Daily Performance Metrics - Aggregated model performance by day/site/device
-- ================================================================

CREATE TABLE IF NOT EXISTS pms_metrics_daily (
    id BIGSERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    site_id VARCHAR(50) NOT NULL,
    device_family VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,

    -- Sample counts
    n_predictions INTEGER NOT NULL DEFAULT 0,
    n_ground_truth INTEGER NOT NULL DEFAULT 0,
    n_paired_cases INTEGER NOT NULL DEFAULT 0, -- predictions with ground truth

    -- Classification performance metrics
    balanced_accuracy NUMERIC(5,4), -- Primary endpoint for regulatory reporting
    auc_roc NUMERIC(5,4),
    auc_pr NUMERIC(5,4),
    sensitivity NUMERIC(5,4), -- True Positive Rate
    specificity NUMERIC(5,4), -- True Negative Rate
    precision NUMERIC(5,4),
    f1_score NUMERIC(5,4),

    -- Calibration and confidence metrics
    expected_calibration_error NUMERIC(5,4), -- ECE for regulatory calibration assessment
    brier_score NUMERIC(5,4),
    mean_confidence NUMERIC(5,4),
    confidence_variance NUMERIC(5,4),

    -- Data quality metrics
    usable_data_percentage NUMERIC(5,2) NOT NULL DEFAULT 0.0, -- % sessions meeting QC thresholds
    mean_signal_quality NUMERIC(5,4),
    artifact_rate NUMERIC(5,4), -- % time with artifacts

    -- Baseline comparison (vs training performance)
    ba_delta_vs_training NUMERIC(6,4), -- Change from training BA
    auc_delta_vs_training NUMERIC(6,4), -- Change from training AUC

    -- Computed at time of creation
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Optimized indexes for metrics queries
CREATE UNIQUE INDEX IF NOT EXISTS idx_pms_metrics_unique ON pms_metrics_daily (metric_date, site_id, device_family, model_version);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_metrics_date_desc ON pms_metrics_daily (metric_date DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_metrics_site_date ON pms_metrics_daily (site_id, metric_date DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_metrics_model_date ON pms_metrics_daily (model_version, metric_date DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pms_metrics_ba ON pms_metrics_daily (balanced_accuracy) WHERE balanced_accuracy IS NOT NULL;

-- ================================================================
-- Model Drift Detection - Daily drift statistics and alerts
-- ================================================================

CREATE TABLE IF NOT EXISTS model_drift_daily (
    id BIGSERIAL PRIMARY KEY,
    drift_date DATE NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    site_id VARCHAR(50), -- NULL for global drift

    -- Population shift detection (feature drift)
    psi_features_json JSONB, -- Population Stability Index per feature
    max_psi NUMERIC(6,4), -- Highest PSI across all features
    high_drift_features TEXT[], -- Features with PSI > 0.2

    -- Performance drift detection (CUSUM)
    cusum_ba NUMERIC(8,4), -- Cumulative sum of BA deviations
    cusum_auc NUMERIC(8,4), -- Cumulative sum of AUC deviations
    cusum_threshold_ba NUMERIC(6,4) DEFAULT 0.05, -- Threshold for BA drift alert
    cusum_threshold_auc NUMERIC(6,4) DEFAULT 0.05, -- Threshold for AUC drift alert

    -- Calibration drift
    ece_trend NUMERIC(6,4), -- 7-day moving average ECE
    ece_baseline NUMERIC(6,4), -- 30-day baseline ECE
    ece_drift_magnitude NUMERIC(6,4), -- |trend - baseline|

    -- Data volume and quality trends
    daily_volume INTEGER NOT NULL DEFAULT 0,
    volume_7d_avg NUMERIC(8,2), -- 7-day average volume
    volume_30d_avg NUMERIC(8,2), -- 30-day baseline volume
    quality_trend NUMERIC(5,4), -- 7-day moving average quality

    -- Drift status assessment
    drift_status VARCHAR(20) NOT NULL DEFAULT 'stable', -- 'stable', 'warning', 'alert', 'critical'
    drift_confidence NUMERIC(5,4), -- Confidence in drift detection
    requires_investigation BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for drift monitoring
CREATE UNIQUE INDEX IF NOT EXISTS idx_drift_unique ON model_drift_daily (drift_date, model_version, COALESCE(site_id, 'GLOBAL'));
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_drift_date_status ON model_drift_daily (drift_date DESC, drift_status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_drift_model_status ON model_drift_daily (model_version, drift_status, drift_date DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_drift_psi_gin ON model_drift_daily USING GIN (psi_features_json);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_drift_investigation ON model_drift_daily (requires_investigation, drift_date DESC) WHERE requires_investigation = TRUE;

-- ================================================================
-- Safety Alerts - Automated and manual safety alerts with escalation
-- ================================================================

CREATE TABLE IF NOT EXISTS safety_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type VARCHAR(50) NOT NULL, -- 'performance_drop', 'calibration_drift', 'data_quality', 'adverse_event', 'manual'
    severity VARCHAR(20) NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    rule_id VARCHAR(100) NOT NULL, -- ID of rule that triggered alert

    -- Alert context
    site_id VARCHAR(50), -- NULL for global alerts
    model_version VARCHAR(20),
    device_family VARCHAR(50),
    subject_id VARCHAR(50), -- For subject-specific alerts

    -- Alert lifecycle
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    closed_at TIMESTAMPTZ,
    closed_by VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'open', -- 'open', 'acknowledged', 'investigating', 'resolved', 'false_positive'

    -- Alert content
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    evidence_json JSONB, -- Supporting data for the alert
    evidence_zip_path VARCHAR(500), -- Path to detailed evidence package

    -- Escalation tracking
    escalation_level INTEGER DEFAULT 0, -- 0=initial, 1=L1 escalation, 2=L2, etc.
    escalated_at TIMESTAMPTZ,
    escalation_reason TEXT,
    next_escalation_due TIMESTAMPTZ,

    -- Resolution tracking
    root_cause TEXT,
    corrective_actions TEXT[],
    preventive_actions TEXT[],
    impact_assessment TEXT,

    -- Regulatory requirements
    regulatory_notification_required BOOLEAN DEFAULT FALSE,
    regulatory_notified_at TIMESTAMPTZ,
    regulatory_reference VARCHAR(100),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Safety alerts indexes for rapid response
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_safety_alerts_status_severity ON safety_alerts (status, severity, opened_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_safety_alerts_site_open ON safety_alerts (site_id, status, opened_at DESC) WHERE status IN ('open', 'acknowledged');
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_safety_alerts_escalation ON safety_alerts (next_escalation_due) WHERE next_escalation_due IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_safety_alerts_regulatory ON safety_alerts (regulatory_notification_required, regulatory_notified_at) WHERE regulatory_notification_required = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_safety_alerts_evidence_gin ON safety_alerts USING GIN (evidence_json);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_safety_alerts_rule ON safety_alerts (rule_id, opened_at DESC);

-- ================================================================
-- Alert Rules Configuration - Configurable rules for automated alerting
-- ================================================================

CREATE TABLE IF NOT EXISTS alert_rules (
    rule_id VARCHAR(100) PRIMARY KEY,
    rule_name VARCHAR(200) NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- 'performance', 'drift', 'quality', 'volume', 'custom'

    -- Rule parameters
    metric_name VARCHAR(100) NOT NULL, -- e.g., 'balanced_accuracy', 'psi_max', 'usable_data_percentage'
    threshold_value NUMERIC(10,6) NOT NULL,
    comparison_operator VARCHAR(10) NOT NULL, -- '<', '>', '<=', '>=', '!=', '='
    lookback_days INTEGER DEFAULT 7,
    baseline_days INTEGER DEFAULT 30,

    -- Scope
    applies_to_sites TEXT[], -- NULL = all sites
    applies_to_models TEXT[], -- NULL = all models
    applies_to_devices TEXT[], -- NULL = all devices

    -- Alert configuration
    severity VARCHAR(20) NOT NULL,
    title_template VARCHAR(200) NOT NULL,
    description_template TEXT NOT NULL,

    -- Escalation settings
    escalation_enabled BOOLEAN DEFAULT TRUE,
    escalation_hours INTEGER DEFAULT 24,
    max_escalation_level INTEGER DEFAULT 2,

    -- Rule lifecycle
    enabled BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by VARCHAR(100),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_triggered_at TIMESTAMPTZ
);

-- Alert rules indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_rules_enabled ON alert_rules (enabled, rule_type) WHERE enabled = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_rules_metric ON alert_rules (metric_name, enabled);

-- ================================================================
-- Model Registry - Track deployed models and their metadata
-- ================================================================

CREATE TABLE IF NOT EXISTS model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(20) UNIQUE NOT NULL,
    model_name VARCHAR(100) NOT NULL,

    -- Model metadata
    algorithm_type VARCHAR(50) NOT NULL, -- 'random_forest', 'logistic_regression', 'neural_network'
    training_dataset_id VARCHAR(100),
    training_completed_at TIMESTAMPTZ,

    -- Performance baselines (from training/validation)
    baseline_balanced_accuracy NUMERIC(5,4) NOT NULL,
    baseline_auc_roc NUMERIC(5,4) NOT NULL,
    baseline_sensitivity NUMERIC(5,4) NOT NULL,
    baseline_specificity NUMERIC(5,4) NOT NULL,
    baseline_ece NUMERIC(5,4),

    -- Deployment tracking
    deployment_status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 'pending', 'deployed', 'deprecated', 'retired'
    deployed_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    retirement_scheduled_at TIMESTAMPTZ,

    -- Feature schema and preprocessing
    feature_schema_json JSONB NOT NULL, -- Schema for drift detection
    preprocessing_config_json JSONB,

    -- Model artifacts
    model_file_path VARCHAR(500),
    model_checksum VARCHAR(64), -- SHA-256 for integrity verification
    docker_image VARCHAR(200),

    -- Regulatory information
    regulatory_approval_status VARCHAR(50),
    fda_510k_number VARCHAR(20),
    ce_mark_number VARCHAR(20),
    clinical_evidence_path VARCHAR(500),

    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model registry indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_registry_status ON model_registry (deployment_status, deployed_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_registry_version ON model_registry (model_version);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_registry_schema_gin ON model_registry USING GIN (feature_schema_json);

-- ================================================================
-- Site Registry - Track deployment sites and their characteristics
-- ================================================================

CREATE TABLE IF NOT EXISTS site_registry (
    site_id VARCHAR(50) PRIMARY KEY,
    site_name VARCHAR(200) NOT NULL,
    site_type VARCHAR(50) NOT NULL, -- 'hospital', 'clinic', 'telehealth', 'research'

    -- Geographic information
    country_code CHAR(2) NOT NULL,
    region VARCHAR(100),
    timezone VARCHAR(50) NOT NULL,

    -- Technical capabilities
    supported_devices TEXT[] NOT NULL,
    bandwidth_category VARCHAR(20), -- 'low', 'medium', 'high'
    technical_support_level VARCHAR(20), -- 'basic', 'standard', 'premium'

    -- Patient population characteristics (for drift context)
    typical_age_range VARCHAR(20),
    typical_disease_severity VARCHAR(20),
    population_size_category VARCHAR(20), -- 'small', 'medium', 'large'

    -- Quality and compliance
    data_quality_tier VARCHAR(20), -- 'tier1', 'tier2', 'tier3'
    regulatory_status VARCHAR(50),
    last_audit_date DATE,

    -- Contact information
    primary_contact_name VARCHAR(100),
    primary_contact_email VARCHAR(100),
    emergency_contact_phone VARCHAR(20),

    -- Site lifecycle
    activation_date DATE NOT NULL,
    deactivation_date DATE,
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- 'active', 'suspended', 'inactive'

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Site registry indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_site_registry_status ON site_registry (status, activation_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_site_registry_country ON site_registry (country_code, status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_site_registry_devices_gin ON site_registry USING GIN (supported_devices);

-- ================================================================
-- Regulatory Reporting - Track regulatory submissions and responses
-- ================================================================

CREATE TABLE IF NOT EXISTS regulatory_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(50) NOT NULL, -- 'PSUR', 'PMSR', 'annual_report', 'safety_update', 'incident_report'

    -- Reporting period
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Regulatory context
    regulatory_authority VARCHAR(50) NOT NULL, -- 'FDA', 'EMA', 'Health_Canada', etc.
    submission_reference VARCHAR(100),

    -- Report content
    title VARCHAR(200) NOT NULL,
    executive_summary TEXT,
    safety_summary TEXT,
    performance_summary TEXT,

    -- Data aggregation
    total_subjects INTEGER,
    total_sessions INTEGER,
    total_predictions INTEGER,
    adverse_events_count INTEGER,
    serious_adverse_events_count INTEGER,

    -- Key metrics for period
    overall_balanced_accuracy NUMERIC(5,4),
    overall_auc_roc NUMERIC(5,4),
    data_quality_score NUMERIC(5,4),

    -- Report artifacts
    report_file_path VARCHAR(500),
    supporting_data_path VARCHAR(500),
    statistical_analysis_path VARCHAR(500),

    -- Submission tracking
    status VARCHAR(20) NOT NULL DEFAULT 'draft', -- 'draft', 'submitted', 'acknowledged', 'approved', 'rejected'
    submitted_at TIMESTAMPTZ,
    submitted_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    authority_response TEXT,

    -- Follow-up actions
    follow_up_required BOOLEAN DEFAULT FALSE,
    follow_up_due_date DATE,
    follow_up_completed_at TIMESTAMPTZ,

    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Regulatory reports indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_regulatory_reports_type_period ON regulatory_reports (report_type, period_end DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_regulatory_reports_authority ON regulatory_reports (regulatory_authority, status, period_end DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_regulatory_reports_followup ON regulatory_reports (follow_up_required, follow_up_due_date) WHERE follow_up_required = TRUE;

-- ================================================================
-- Performance Views - Optimized views for common queries
-- ================================================================

-- Real-time performance summary
CREATE OR REPLACE VIEW pms_performance_summary AS
SELECT
    m.site_id,
    m.model_version,
    m.device_family,
    m.metric_date,
    m.balanced_accuracy,
    m.auc_roc,
    m.expected_calibration_error,
    m.usable_data_percentage,
    m.n_paired_cases,

    -- 7-day moving averages
    AVG(m.balanced_accuracy) OVER (
        PARTITION BY m.site_id, m.model_version, m.device_family
        ORDER BY m.metric_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ba_7d_avg,

    AVG(m.usable_data_percentage) OVER (
        PARTITION BY m.site_id, m.model_version, m.device_family
        ORDER BY m.metric_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS quality_7d_avg,

    -- Performance vs baseline
    r.baseline_balanced_accuracy,
    (m.balanced_accuracy - r.baseline_balanced_accuracy) AS ba_delta_baseline,

    -- Alert indicators
    CASE
        WHEN m.balanced_accuracy < r.baseline_balanced_accuracy - 0.10 THEN 'CRITICAL'
        WHEN m.balanced_accuracy < r.baseline_balanced_accuracy - 0.05 THEN 'HIGH'
        WHEN m.expected_calibration_error > 0.08 THEN 'MEDIUM'
        WHEN m.usable_data_percentage < 80 THEN 'HIGH'
        ELSE 'OK'
    END AS alert_level

FROM pms_metrics_daily m
JOIN model_registry r ON m.model_version = r.model_version
WHERE m.metric_date >= CURRENT_DATE - INTERVAL '30 days';

-- Active alerts summary
CREATE OR REPLACE VIEW active_alerts_summary AS
SELECT
    sa.severity,
    sa.alert_type,
    sa.site_id,
    sa.model_version,
    COUNT(*) as alert_count,
    MIN(sa.opened_at) as oldest_alert,
    COUNT(*) FILTER (WHERE sa.status = 'open') as open_count,
    COUNT(*) FILTER (WHERE sa.status = 'acknowledged') as acknowledged_count,
    COUNT(*) FILTER (WHERE sa.escalation_level > 0) as escalated_count
FROM safety_alerts sa
WHERE sa.status IN ('open', 'acknowledged', 'investigating')
GROUP BY sa.severity, sa.alert_type, sa.site_id, sa.model_version
ORDER BY
    CASE sa.severity
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        WHEN 'MEDIUM' THEN 3
        ELSE 4
    END,
    oldest_alert;

-- Drift monitoring summary
CREATE OR REPLACE VIEW drift_monitoring_summary AS
SELECT
    d.model_version,
    d.site_id,
    d.drift_date,
    d.max_psi,
    d.cusum_ba,
    d.ece_drift_magnitude,
    d.drift_status,
    d.drift_confidence,

    -- Trend indicators
    LAG(d.max_psi, 1) OVER (PARTITION BY d.model_version, d.site_id ORDER BY d.drift_date) AS prev_max_psi,
    LAG(d.cusum_ba, 1) OVER (PARTITION BY d.model_version, d.site_id ORDER BY d.drift_date) AS prev_cusum_ba,

    -- Risk scoring
    CASE
        WHEN d.max_psi > 0.25 OR d.cusum_ba > 0.10 THEN 'HIGH_RISK'
        WHEN d.max_psi > 0.15 OR d.cusum_ba > 0.05 THEN 'MEDIUM_RISK'
        WHEN d.max_psi > 0.10 OR d.cusum_ba > 0.02 THEN 'LOW_RISK'
        ELSE 'STABLE'
    END AS risk_level

FROM model_drift_daily d
WHERE d.drift_date >= CURRENT_DATE - INTERVAL '14 days'
ORDER BY d.drift_date DESC, d.drift_confidence DESC;

-- ================================================================
-- Triggers for automated updates
-- ================================================================

-- Update pms_metrics_daily updated_at on changes
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_pms_metrics_daily_updated_at
    BEFORE UPDATE ON pms_metrics_daily
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_safety_alerts_updated_at
    BEFORE UPDATE ON safety_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_registry_updated_at
    BEFORE UPDATE ON model_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ================================================================
-- Initial Alert Rules - Starter set of monitoring rules
-- ================================================================

INSERT INTO alert_rules (rule_id, rule_name, rule_type, metric_name, threshold_value, comparison_operator,
                        lookback_days, severity, title_template, description_template, created_by) VALUES

-- Performance monitoring rules
('PERF_BA_DROP_10PCT', 'Balanced Accuracy Drop >10%', 'performance', 'balanced_accuracy', -0.10, '<', 7, 'HIGH',
 'Performance Alert: BA Drop >10% at {site_id}',
 'Balanced accuracy has dropped by more than 10% compared to 30-day baseline at site {site_id} for model {model_version}', 'system'),

('PERF_BA_DROP_5PCT', 'Balanced Accuracy Drop >5%', 'performance', 'balanced_accuracy', -0.05, '<', 7, 'MEDIUM',
 'Performance Warning: BA Drop >5% at {site_id}',
 'Balanced accuracy has dropped by more than 5% compared to 30-day baseline at site {site_id} for model {model_version}', 'system'),

-- Calibration monitoring rules
('CALIB_ECE_HIGH', 'Expected Calibration Error >8%', 'performance', 'expected_calibration_error', 0.08, '>', 7, 'MEDIUM',
 'Calibration Alert: ECE >8% at {site_id}',
 'Expected calibration error exceeded 8% threshold at site {site_id}, indicating poor model calibration', 'system'),

-- Data quality monitoring rules
('QUALITY_LOW_USABLE', 'Usable Data <80%', 'quality', 'usable_data_percentage', 80.0, '<', 7, 'HIGH',
 'Data Quality Alert: <80% Usable at {site_id}',
 'Usable data percentage dropped below 80% at site {site_id}, indicating significant quality issues', 'system'),

('QUALITY_LOW_USABLE_WARN', 'Usable Data <90%', 'quality', 'usable_data_percentage', 90.0, '<', 7, 'MEDIUM',
 'Data Quality Warning: <90% Usable at {site_id}',
 'Usable data percentage dropped below 90% at site {site_id}, monitoring for quality degradation', 'system'),

-- Drift monitoring rules
('DRIFT_PSI_HIGH', 'Population Shift PSI >0.2', 'drift', 'max_psi', 0.20, '>', 1, 'MEDIUM',
 'Population Drift Alert: PSI >0.2 for {model_version}',
 'Population Stability Index exceeded 0.2 for model {model_version}, indicating significant population shift', 'system'),

('DRIFT_PSI_CRITICAL', 'Population Shift PSI >0.3', 'drift', 'max_psi', 0.30, '>', 1, 'HIGH',
 'Critical Population Drift: PSI >0.3 for {model_version}',
 'Population Stability Index exceeded 0.3 for model {model_version}, requiring immediate investigation', 'system'),

-- Volume monitoring rules
('VOLUME_DROP_50PCT', 'Daily Volume Drop >50%', 'volume', 'daily_volume', -0.50, '<', 1, 'HIGH',
 'Volume Alert: >50% Drop at {site_id}',
 'Daily prediction volume dropped by more than 50% at site {site_id}, indicating potential system issues', 'system');

-- ================================================================
-- Comments and Documentation
-- ================================================================

COMMENT ON TABLE pms_events IS 'Central event log for all post-market surveillance activities including predictions, ground truth updates, and adverse events';
COMMENT ON TABLE pms_metrics_daily IS 'Daily aggregated performance metrics by site/device/model for regulatory reporting and trend analysis';
COMMENT ON TABLE model_drift_daily IS 'Daily model drift statistics including population shift (PSI) and performance drift (CUSUM) detection';
COMMENT ON TABLE safety_alerts IS 'Automated and manual safety alerts with full escalation and resolution tracking for regulatory compliance';
COMMENT ON TABLE alert_rules IS 'Configurable alerting rules for automated safety monitoring with flexible thresholds and scoping';
COMMENT ON TABLE model_registry IS 'Registry of deployed models with baseline performance metrics and regulatory approval status';
COMMENT ON TABLE site_registry IS 'Registry of deployment sites with capabilities and population characteristics for drift context';
COMMENT ON TABLE regulatory_reports IS 'Tracking of regulatory submissions including PSURs, PMSRs, and safety updates with authority responses';

-- Schema version for migration tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description) VALUES
('1.0.0', 'Initial PMS schema with events, metrics, drift detection, and safety alerting');

-- ================================================================
-- Performance optimization hints
-- ================================================================

-- For high-volume ingestion, consider partitioning pms_events by date
-- ALTER TABLE pms_events PARTITION BY RANGE (event_date);

-- For large deployments, consider read replicas for analytical queries
-- while keeping writes on primary for real-time alerting

-- Consider materialized views for complex aggregations if query performance
-- becomes an issue with large data volumes

-- ================================================================
-- Grants and Security
-- ================================================================

-- Example role-based security (adjust based on your user management)
-- CREATE ROLE pms_ingest_role;
-- CREATE ROLE pms_analyst_role;
-- CREATE ROLE pms_admin_role;

-- GRANT INSERT ON pms_events TO pms_ingest_role;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO pms_analyst_role;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pms_admin_role;