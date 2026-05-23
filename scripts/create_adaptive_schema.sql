-- Adaptive Clinical AI Database Schema
-- Tables for closed-loop neurofeedback sessions with audit trails

-- ====================================
-- ADAPTIVE SESSIONS MANAGEMENT
-- ====================================

CREATE TABLE IF NOT EXISTS adaptive_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trial_id VARCHAR(50) NOT NULL,
    subject_id VARCHAR(50) NOT NULL,
    site_id VARCHAR(50) NOT NULL,
    clinician_id VARCHAR(50) NOT NULL,
    mode VARCHAR(20) NOT NULL CHECK (mode IN ('shadow', 'assist', 'active')),
    target_biomarker VARCHAR(50) NOT NULL,
    target_value FLOAT NOT NULL CHECK (target_value BETWEEN 0.1 AND 1.0),
    session_duration_minutes INTEGER NOT NULL CHECK (session_duration_minutes BETWEEN 5 AND 60),
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'terminated')),
    total_ticks INTEGER DEFAULT 0,
    safety_violations INTEGER DEFAULT 0,
    avg_confidence FLOAT DEFAULT 0.0,
    dose_used FLOAT DEFAULT 0.0,
    config_json JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for adaptive sessions
CREATE INDEX IF NOT EXISTS idx_adaptive_sessions_trial_id ON adaptive_sessions(trial_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_sessions_subject_id ON adaptive_sessions(subject_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_sessions_site_id ON adaptive_sessions(site_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_sessions_clinician_id ON adaptive_sessions(clinician_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_sessions_status ON adaptive_sessions(status);
CREATE INDEX IF NOT EXISTS idx_adaptive_sessions_started_at ON adaptive_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_adaptive_sessions_mode ON adaptive_sessions(mode);

-- ====================================
-- CONTROL TICKS AUDIT TRAIL
-- ====================================

CREATE TABLE IF NOT EXISTS adaptive_ticks (
    tick_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES adaptive_sessions(session_id) ON DELETE CASCADE,
    tick_number INTEGER NOT NULL,
    timestamp_tick TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Input features
    features_json JSONB NOT NULL,
    model_confidence FLOAT NOT NULL CHECK (model_confidence BETWEEN 0.0 AND 1.0),
    qc_json JSONB NOT NULL,
    vitals_json JSONB NOT NULL,
    previous_feedback_json JSONB NOT NULL,

    -- Controller output
    output_gain FLOAT NOT NULL CHECK (output_gain BETWEEN 0.1 AND 2.0),
    output_threshold FLOAT NOT NULL CHECK (output_threshold BETWEEN 0.1 AND 1.0),
    output_frequency_band VARCHAR(20) NOT NULL DEFAULT 'beta',
    output_modulation_type VARCHAR(20) NOT NULL DEFAULT 'amplitude',

    -- Safety and decision metadata
    safety_state VARCHAR(20) NOT NULL,
    gates_pass BOOLEAN NOT NULL DEFAULT TRUE,
    guards_json JSONB NOT NULL,
    rationale TEXT NOT NULL,
    confidence_weight FLOAT NOT NULL DEFAULT 0.0,
    dose_increment FLOAT NOT NULL DEFAULT 0.0,
    cumulative_dose FLOAT NOT NULL DEFAULT 0.0,

    -- Decision factors (for explainability)
    control_error FLOAT DEFAULT 0.0,
    pid_proportional FLOAT DEFAULT 0.0,
    pid_integral FLOAT DEFAULT 0.0,
    pid_derivative FLOAT DEFAULT 0.0,
    raw_action FLOAT DEFAULT 0.0,
    weighted_action FLOAT DEFAULT 0.0,
    biased_action FLOAT DEFAULT 0.0,
    drift_bias_factor FLOAT DEFAULT 1.0,
    safety_projection_applied BOOLEAN DEFAULT FALSE,

    -- Metadata
    decision_metadata_json JSONB,
    pms_event_sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for adaptive ticks
CREATE INDEX IF NOT EXISTS idx_adaptive_ticks_session_id ON adaptive_ticks(session_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_ticks_timestamp ON adaptive_ticks(timestamp_tick);
CREATE INDEX IF NOT EXISTS idx_adaptive_ticks_session_tick ON adaptive_ticks(session_id, tick_number);
CREATE INDEX IF NOT EXISTS idx_adaptive_ticks_safety_state ON adaptive_ticks(safety_state);
CREATE INDEX IF NOT EXISTS idx_adaptive_ticks_gates_pass ON adaptive_ticks(gates_pass);
CREATE INDEX IF NOT EXISTS idx_adaptive_ticks_model_confidence ON adaptive_ticks(model_confidence);

-- ====================================
-- POLICY VERSIONS AND VALIDATION
-- ====================================

CREATE TABLE IF NOT EXISTS adaptive_policy_versions (
    version VARCHAR(20) PRIMARY KEY,
    policy_hash VARCHAR(64) NOT NULL UNIQUE,
    params_json JSONB NOT NULL,
    safety_limits_json JSONB NOT NULL,
    controller_config_json JSONB NOT NULL,

    -- Validation status
    validation_status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (validation_status IN ('pending', 'validated', 'rejected', 'deprecated')),
    validation_report_path TEXT,
    validation_summary_json JSONB,

    -- Approval workflow
    approved_by VARCHAR(50),
    approved_at TIMESTAMP WITH TIME ZONE,
    approval_notes TEXT,

    -- Deployment tracking
    deployment_sites TEXT[],
    deployed_at TIMESTAMP WITH TIME ZONE,
    deprecated_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for policy versions
CREATE INDEX IF NOT EXISTS idx_adaptive_policy_versions_status ON adaptive_policy_versions(validation_status);
CREATE INDEX IF NOT EXISTS idx_adaptive_policy_versions_approved ON adaptive_policy_versions(approved_at);
CREATE INDEX IF NOT EXISTS idx_adaptive_policy_versions_deployed ON adaptive_policy_versions(deployed_at);

-- ====================================
-- SUPERVISOR OVERRIDES
-- ====================================

CREATE TABLE IF NOT EXISTS adaptive_overrides (
    override_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES adaptive_sessions(session_id) ON DELETE CASCADE,
    override_action VARCHAR(20) NOT NULL
        CHECK (override_action IN ('pause', 'resume', 'rollback', 'emergency_stop')),
    override_reason TEXT,
    override_user VARCHAR(50) NOT NULL,
    override_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Override status
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'cleared', 'expired')),
    cleared_by VARCHAR(50),
    cleared_at TIMESTAMP WITH TIME ZONE,
    cooldown_until TIMESTAMP WITH TIME ZONE,

    -- Context when override was applied
    tick_number_at_override INTEGER,
    safety_state_at_override VARCHAR(20),
    feedback_params_json JSONB,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for overrides
CREATE INDEX IF NOT EXISTS idx_adaptive_overrides_session_id ON adaptive_overrides(session_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_overrides_status ON adaptive_overrides(status);
CREATE INDEX IF NOT EXISTS idx_adaptive_overrides_timestamp ON adaptive_overrides(override_timestamp);
CREATE INDEX IF NOT EXISTS idx_adaptive_overrides_user ON adaptive_overrides(override_user);

-- ====================================
-- SIMULATION AND VALIDATION RESULTS
-- ====================================

CREATE TABLE IF NOT EXISTS adaptive_simulation_runs (
    simulation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_version VARCHAR(20) NOT NULL REFERENCES adaptive_policy_versions(version),
    simulation_type VARCHAR(20) NOT NULL
        CHECK (simulation_type IN ('replay', 'counterfactual', 'monte_carlo', 'stability')),

    -- Input parameters
    input_dataset VARCHAR(100) NOT NULL,
    num_sessions INTEGER NOT NULL,
    num_subjects INTEGER,
    session_duration_minutes INTEGER NOT NULL,
    target_biomarkers TEXT[] NOT NULL,

    -- Results summary
    total_ticks INTEGER NOT NULL DEFAULT 0,
    safety_violations INTEGER NOT NULL DEFAULT 0,
    stability_score FLOAT,
    time_in_green_percentage FLOAT,
    dose_usage_stats_json JSONB,
    adverse_surrogate_rate FLOAT,

    -- Performance metrics
    biomarker_target_achievement_rate FLOAT,
    mean_control_error FLOAT,
    control_stability_metric FLOAT,
    oscillation_detected BOOLEAN DEFAULT FALSE,

    -- Comparison results (if counterfactual)
    baseline_comparison_json JSONB,
    statistical_significance_json JSONB,
    non_inferiority_result BOOLEAN,
    superiority_result BOOLEAN,

    -- Validation outcomes
    validation_passed BOOLEAN DEFAULT FALSE,
    validation_criteria_json JSONB,
    validation_notes TEXT,

    -- Metadata
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_seconds INTEGER,
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for simulation runs
CREATE INDEX IF NOT EXISTS idx_adaptive_simulation_runs_policy ON adaptive_simulation_runs(policy_version);
CREATE INDEX IF NOT EXISTS idx_adaptive_simulation_runs_type ON adaptive_simulation_runs(simulation_type);
CREATE INDEX IF NOT EXISTS idx_adaptive_simulation_runs_validation ON adaptive_simulation_runs(validation_passed);
CREATE INDEX IF NOT EXISTS idx_adaptive_simulation_runs_started ON adaptive_simulation_runs(started_at);

-- ====================================
-- FEATURE FLAGS AND CONFIGURATION
-- ====================================

CREATE TABLE IF NOT EXISTS adaptive_feature_flags (
    flag_name VARCHAR(50) PRIMARY KEY,
    flag_value BOOLEAN NOT NULL DEFAULT FALSE,
    site_id VARCHAR(50),
    trial_id VARCHAR(50),

    -- Configuration
    config_json JSONB,
    description TEXT,

    -- Safety limits (site-specific overrides)
    safety_overrides_json JSONB,
    max_sessions_per_day INTEGER DEFAULT 10,
    auto_pause_on_alert BOOLEAN DEFAULT TRUE,

    -- Rollout controls
    rollout_percentage FLOAT DEFAULT 0.0 CHECK (rollout_percentage BETWEEN 0.0 AND 100.0),
    allowed_modes TEXT[] DEFAULT ARRAY['shadow'],

    -- Metadata
    enabled_by VARCHAR(50),
    enabled_at TIMESTAMP WITH TIME ZONE,
    disabled_by VARCHAR(50),
    disabled_at TIMESTAMP WITH TIME ZONE,
    notes TEXT,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for feature flags
CREATE INDEX IF NOT EXISTS idx_adaptive_feature_flags_site ON adaptive_feature_flags(site_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_feature_flags_trial ON adaptive_feature_flags(trial_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_feature_flags_enabled ON adaptive_feature_flags(flag_value);

-- ====================================
-- SAFETY INCIDENTS AND ESCALATIONS
-- ====================================

CREATE TABLE IF NOT EXISTS adaptive_safety_incidents (
    incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES adaptive_sessions(session_id) ON DELETE CASCADE,
    incident_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),

    -- Incident details
    description TEXT NOT NULL,
    tick_number_at_incident INTEGER,
    safety_state_at_incident VARCHAR(20),
    feedback_params_at_incident_json JSONB,
    vitals_at_incident_json JSONB,

    -- Root cause analysis
    root_cause_analysis TEXT,
    contributing_factors TEXT[],
    system_response_adequate BOOLEAN,

    -- Resolution
    resolution_actions TEXT[],
    resolved_by VARCHAR(50),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,

    -- Regulatory reporting
    reportable_to_fda BOOLEAN DEFAULT FALSE,
    reportable_to_ema BOOLEAN DEFAULT FALSE,
    reported_to_regulators BOOLEAN DEFAULT FALSE,
    regulatory_report_refs TEXT[],

    -- Investigation
    investigation_required BOOLEAN DEFAULT TRUE,
    investigation_assigned_to VARCHAR(50),
    investigation_completed_at TIMESTAMP WITH TIME ZONE,
    investigation_report_path TEXT,

    -- Metadata
    detected_by VARCHAR(20) NOT NULL DEFAULT 'system',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for safety incidents
CREATE INDEX IF NOT EXISTS idx_adaptive_safety_incidents_session ON adaptive_safety_incidents(session_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_safety_incidents_type ON adaptive_safety_incidents(incident_type);
CREATE INDEX IF NOT EXISTS idx_adaptive_safety_incidents_severity ON adaptive_safety_incidents(severity);
CREATE INDEX IF NOT EXISTS idx_adaptive_safety_incidents_reportable ON adaptive_safety_incidents(reportable_to_fda, reportable_to_ema);
CREATE INDEX IF NOT EXISTS idx_adaptive_safety_incidents_created ON adaptive_safety_incidents(created_at);

-- ====================================
-- PERFORMANCE ANALYTICS
-- ====================================

CREATE TABLE IF NOT EXISTS adaptive_daily_analytics (
    analytics_date DATE NOT NULL,
    site_id VARCHAR(50) NOT NULL,
    mode VARCHAR(20) NOT NULL,

    -- Session counts
    total_sessions INTEGER NOT NULL DEFAULT 0,
    completed_sessions INTEGER NOT NULL DEFAULT 0,
    terminated_sessions INTEGER NOT NULL DEFAULT 0,

    -- Safety metrics
    total_safety_violations INTEGER NOT NULL DEFAULT 0,
    total_supervisor_overrides INTEGER NOT NULL DEFAULT 0,
    safety_incidents INTEGER NOT NULL DEFAULT 0,

    -- Performance metrics
    avg_confidence FLOAT DEFAULT 0.0,
    avg_dose_used FLOAT DEFAULT 0.0,
    avg_time_in_adaptive FLOAT DEFAULT 0.0,
    avg_biomarker_improvement FLOAT DEFAULT 0.0,

    -- Controller performance
    avg_control_error FLOAT DEFAULT 0.0,
    stability_score FLOAT DEFAULT 0.0,
    oscillation_rate FLOAT DEFAULT 0.0,

    -- Quality metrics
    avg_signal_quality FLOAT DEFAULT 0.0,
    artifact_rate FLOAT DEFAULT 0.0,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    PRIMARY KEY (analytics_date, site_id, mode)
);

-- Indexes for daily analytics
CREATE INDEX IF NOT EXISTS idx_adaptive_daily_analytics_date ON adaptive_daily_analytics(analytics_date);
CREATE INDEX IF NOT EXISTS idx_adaptive_daily_analytics_site ON adaptive_daily_analytics(site_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_daily_analytics_mode ON adaptive_daily_analytics(mode);

-- ====================================
-- DATA RETENTION AND ARCHIVAL
-- ====================================

-- Create partition tables for adaptive_ticks by month
CREATE TABLE IF NOT EXISTS adaptive_ticks_archive (
    LIKE adaptive_ticks INCLUDING ALL
);

-- ====================================
-- FUNCTIONS AND TRIGGERS
-- ====================================

-- Function to update session statistics
CREATE OR REPLACE FUNCTION update_adaptive_session_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE adaptive_sessions
    SET total_ticks = (
        SELECT COUNT(*) FROM adaptive_ticks
        WHERE session_id = NEW.session_id
    ),
    safety_violations = (
        SELECT COUNT(*) FROM adaptive_ticks
        WHERE session_id = NEW.session_id AND gates_pass = FALSE
    ),
    avg_confidence = (
        SELECT AVG(model_confidence) FROM adaptive_ticks
        WHERE session_id = NEW.session_id
    ),
    dose_used = (
        SELECT COALESCE(MAX(cumulative_dose), 0) FROM adaptive_ticks
        WHERE session_id = NEW.session_id
    ),
    updated_at = NOW()
    WHERE session_id = NEW.session_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update session stats on tick insert
DROP TRIGGER IF EXISTS trg_update_session_stats ON adaptive_ticks;
CREATE TRIGGER trg_update_session_stats
    AFTER INSERT ON adaptive_ticks
    FOR EACH ROW
    EXECUTE FUNCTION update_adaptive_session_stats();

-- Function to automatically archive old ticks
CREATE OR REPLACE FUNCTION archive_old_adaptive_ticks()
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER := 0;
    cutoff_date TIMESTAMP WITH TIME ZONE := NOW() - INTERVAL '90 days';
BEGIN
    -- Move old ticks to archive table
    WITH moved_ticks AS (
        DELETE FROM adaptive_ticks
        WHERE timestamp_tick < cutoff_date
        RETURNING *
    )
    INSERT INTO adaptive_ticks_archive SELECT * FROM moved_ticks;

    GET DIAGNOSTICS archived_count = ROW_COUNT;

    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- INITIAL DATA
-- ====================================

-- Insert default policy version
INSERT INTO adaptive_policy_versions (
    version,
    policy_hash,
    params_json,
    safety_limits_json,
    controller_config_json,
    validation_status,
    created_by
) VALUES (
    'v1.0.0',
    'default_policy_hash_v1_0_0',
    '{
        "kp": 0.5,
        "ki": 0.1,
        "kd": 0.05,
        "target_beta_power": 0.65,
        "target_alpha_beta": 0.8,
        "confidence_weight_threshold": 0.75,
        "exploration_enabled": false,
        "bias_factor_drift_med": 0.8,
        "bias_factor_drift_high": 0.6
    }',
    '{
        "min_confidence": 0.75,
        "max_gain_delta_per_10s": 0.10,
        "min_gain": 0.6,
        "max_gain": 1.4,
        "min_threshold": 0.4,
        "max_threshold": 0.9,
        "max_dose_per_20min": 9.0,
        "max_artifact_rate": 0.15,
        "min_coverage": 85.0,
        "rollback_confidence_threshold": 0.6
    }',
    '{
        "frequency_hz": 2.0,
        "biomarker_window_seconds": 10,
        "quality_check_interval_seconds": 5,
        "dose_calculation_method": "gain_based",
        "safety_gate_order": ["quality", "confidence", "bounds", "rate", "dose", "vitals"]
    }',
    'validated',
    'system'
) ON CONFLICT (version) DO NOTHING;

-- Insert default feature flags
INSERT INTO adaptive_feature_flags (flag_name, flag_value, description, allowed_modes) VALUES
    ('adaptive_ai_enabled', FALSE, 'Master switch for adaptive AI functionality', ARRAY['shadow']),
    ('shadow_mode_enabled', TRUE, 'Enable shadow mode testing', ARRAY['shadow']),
    ('assist_mode_enabled', FALSE, 'Enable assisted mode with clinician approval', ARRAY['shadow', 'assist']),
    ('active_mode_enabled', FALSE, 'Enable fully active adaptive mode', ARRAY['shadow', 'assist', 'active'])
ON CONFLICT (flag_name) DO NOTHING;

-- ====================================
-- COMMENTS
-- ====================================

COMMENT ON TABLE adaptive_sessions IS 'Tracks adaptive neurofeedback sessions with safety and audit capabilities';
COMMENT ON TABLE adaptive_ticks IS 'Detailed audit trail of every control tick with full decision context';
COMMENT ON TABLE adaptive_policy_versions IS 'Version control for adaptive policies with validation workflow';
COMMENT ON TABLE adaptive_overrides IS 'Human supervisor overrides and safety interventions';
COMMENT ON TABLE adaptive_simulation_runs IS 'Validation and testing results for adaptive policies';
COMMENT ON TABLE adaptive_feature_flags IS 'Feature flags and site-specific configuration for controlled rollout';
COMMENT ON TABLE adaptive_safety_incidents IS 'Safety incident tracking and regulatory reporting';
COMMENT ON TABLE adaptive_daily_analytics IS 'Daily aggregated performance and safety metrics';

-- ====================================
-- COMPLETION LOG
-- ====================================

INSERT INTO schema_migration_log (migration_name, applied_at) VALUES
    ('adaptive_ai_schema_v1_0_0', NOW())
ON CONFLICT (migration_name) DO NOTHING;