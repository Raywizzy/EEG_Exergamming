#!/usr/bin/env python3
"""
Database Schema Creation for EEG Platform Persistence
Phase VI: Clinical-grade audit trail and job management
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import json

class DatabaseManager:
    """Manage SQLite database for clinical EEG platform."""

    def __init__(self, db_path="data/eeg_platform.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")

    def create_schema(self):
        """Create all tables for clinical platform."""

        # 1. Trials table - IRB/protocol metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                trial_name TEXT NOT NULL,
                protocol_version TEXT NOT NULL,
                irb_number TEXT,
                pi_name TEXT,
                institution TEXT,
                start_date DATE,
                end_date DATE,
                status TEXT CHECK(status IN ('planning', 'active', 'suspended', 'completed', 'terminated')),
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Sites table - Multi-site management
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sites (
                site_id TEXT PRIMARY KEY,
                site_name TEXT NOT NULL,
                institution TEXT NOT NULL,
                country TEXT NOT NULL,
                pi_name TEXT,
                coordinator_email TEXT,
                status TEXT CHECK(status IN ('pending', 'active', 'suspended', 'closed')),
                certification_date DATE,
                last_audit_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. Trial-Site junction table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trial_sites (
                trial_id TEXT,
                site_id TEXT,
                enrollment_target INTEGER,
                enrollment_actual INTEGER DEFAULT 0,
                first_enrollment DATE,
                last_enrollment DATE,
                status TEXT CHECK(status IN ('pending', 'active', 'completed', 'terminated')),
                PRIMARY KEY (trial_id, site_id),
                FOREIGN KEY (trial_id) REFERENCES trials (trial_id),
                FOREIGN KEY (site_id) REFERENCES sites (site_id)
            )
        """)

        # 4. Subjects table - De-identified participant tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS subjects (
                subject_id TEXT PRIMARY KEY,
                trial_id TEXT NOT NULL,
                site_id TEXT NOT NULL,
                screening_number TEXT,
                enrollment_date DATE,
                group_assignment TEXT CHECK(group_assignment IN ('PD', 'CONTROL', 'UNKNOWN')),
                age_group TEXT CHECK(age_group IN ('18-30', '31-45', '46-60', '61-75', '76+')),
                gender TEXT CHECK(gender IN ('M', 'F', 'OTHER', 'UNKNOWN')),
                status TEXT CHECK(status IN ('screened', 'enrolled', 'completed', 'withdrawn', 'excluded')),
                withdrawal_reason TEXT,
                completion_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trial_id) REFERENCES trials (trial_id),
                FOREIGN KEY (site_id) REFERENCES sites (site_id)
            )
        """)

        # 5. Jobs table - EEG processing job tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                subject_id TEXT,
                trial_id TEXT,
                site_id TEXT,
                session_name TEXT,
                job_type TEXT CHECK(job_type IN ('feature_extraction', 'classification', 'qc_check', 'robustness_test')),
                feature_set TEXT CHECK(feature_set IN ('Core5', 'Core15', 'Core15+', 'Custom')),
                status TEXT CHECK(status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
                priority INTEGER DEFAULT 0,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds INTEGER,
                error_message TEXT,
                input_file_path TEXT,
                output_file_path TEXT,
                config_json TEXT,
                FOREIGN KEY (subject_id) REFERENCES subjects (subject_id),
                FOREIGN KEY (trial_id) REFERENCES trials (trial_id),
                FOREIGN KEY (site_id) REFERENCES sites (site_id)
            )
        """)

        # 6. Results table - Classification and QC results
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                result_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                result_type TEXT CHECK(result_type IN ('classification', 'qc_metrics', 'features', 'robustness')),
                prediction TEXT CHECK(prediction IN ('PD', 'CONTROL', 'UNCERTAIN', 'ERROR')),
                confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
                balanced_accuracy REAL CHECK(balanced_accuracy >= 0 AND balanced_accuracy <= 1),
                sensitivity REAL CHECK(sensitivity >= 0 AND sensitivity <= 1),
                specificity REAL CHECK(specificity >= 0 AND specificity <= 1),
                auc REAL CHECK(auc >= 0 AND auc <= 1),
                feature_count INTEGER,
                qc_pass BOOLEAN,
                qc_score REAL CHECK(qc_score >= 0 AND qc_score <= 1),
                artifacts_detected INTEGER DEFAULT 0,
                signal_quality TEXT CHECK(signal_quality IN ('excellent', 'good', 'fair', 'poor', 'failed')),
                results_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            )
        """)

        # 7. Audit Events table - Complete audit trail
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT CHECK(event_type IN ('job_submit', 'job_start', 'job_complete', 'job_fail',
                                                   'user_login', 'data_access', 'config_change', 'system_event',
                                                   'trial_create', 'trial_status_change', 'site_register', 'site_create', 'site_certification',
                                                   'subject_enroll', 'protocol_deviation', 'result_save', 'job_update',
                                                   'model_register', 'model_status_change', 'model_deploy', 'model_load',
                                                   'training_start', 'training_complete', 'model_comparison',
                                                   'explanation_generated', 'explanation_accessed', 'explanation_exported',
                                                   'explanation_modified', 'safety_check_failed', 'data_integrity_violation',
                                                   'threshold_violation', 'model_drift_detected', 'unauthorized_access')),
                entity_type TEXT CHECK(entity_type IN ('job', 'subject', 'trial', 'site', 'user', 'system', 'result', 'audit', 'dashboard', 'model', 'training_run', 'comparison', 'explanation', 'explanation_audit', 'safety_violation')),
                entity_id TEXT,
                user_id TEXT,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                success BOOLEAN DEFAULT TRUE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 8. Feature Metadata table - Track extracted features
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL,
                feature_type TEXT CHECK(feature_type IN ('spectral', 'temporal', 'connectivity', 'stability', 'burst')),
                channel_group TEXT CHECK(channel_group IN ('motor', 'frontal', 'posterior', 'global')),
                frequency_band TEXT CHECK(frequency_band IN ('theta', 'alpha', 'beta', 'gamma', 'broadband')),
                unit TEXT,
                quality_flag BOOLEAN DEFAULT TRUE,
                extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            )
        """)

        # 9. System Configuration table - Platform settings
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                config_id TEXT PRIMARY KEY,
                config_category TEXT CHECK(config_category IN ('processing', 'quality', 'security', 'notification', 'trial_metadata', 'site_metadata', 'subject_metadata', 'monitoring', 'deep_learning', 'explainability')),
                config_key TEXT NOT NULL,
                config_value TEXT NOT NULL,
                config_type TEXT CHECK(config_type IN ('string', 'integer', 'float', 'boolean', 'json')),
                description TEXT,
                is_sensitive BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        """)


        # 10. Model Registry table - Deep learning model management
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_type TEXT CHECK(model_type IN ('core15+', 'cnn_spectrogram', 'rnn_temporal', 'transformer', 'ensemble')),
                architecture TEXT NOT NULL,
                version TEXT NOT NULL,
                framework TEXT CHECK(framework IN ('sklearn', 'pytorch', 'tensorflow', 'custom')),
                hyperparameters TEXT, -- JSON
                training_data_sources TEXT, -- JSON array of dataset names
                performance_metrics TEXT, -- JSON
                model_path TEXT NOT NULL,
                checkpoint_path TEXT,
                is_deployed BOOLEAN DEFAULT FALSE,
                deployment_sites TEXT, -- JSON array of site_ids
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT CHECK(status IN ('training', 'validation', 'deployed', 'deprecated', 'failed'))
            )
        """)

        # 11. Training Runs table - Track model training sessions
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                experiment_name TEXT,
                run_name TEXT,
                training_config TEXT, -- JSON
                dataset_split TEXT, -- JSON
                training_metrics TEXT, -- JSON (loss, accuracy over epochs)
                validation_metrics TEXT, -- JSON
                final_performance TEXT, -- JSON
                training_duration_seconds INTEGER,
                gpu_used BOOLEAN DEFAULT FALSE,
                gpu_model TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
                error_message TEXT,
                artifacts_path TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)

        # 12. Model Predictions table - Track predictions from all models
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                prediction_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                subject_id TEXT,
                prediction_class TEXT CHECK(prediction_class IN ('PD', 'CONTROL', 'UNCERTAIN', 'ERROR')),
                confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
                class_probabilities TEXT, -- JSON object with class probabilities
                feature_importance TEXT, -- JSON for explainability
                processing_time_ms INTEGER,
                input_shape TEXT, -- JSON describing input dimensions
                model_version TEXT,
                prediction_metadata TEXT, -- JSON for additional model-specific data
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs (job_id),
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)

        # 13. Model Comparisons table - Track head-to-head model performance
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_comparisons (
                comparison_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                model_ids TEXT NOT NULL, -- JSON array of model_ids being compared
                dataset_name TEXT NOT NULL,
                validation_type TEXT CHECK(validation_type IN ('loso', 'kfold', 'holdout', 'temporal')),
                comparison_metrics TEXT, -- JSON with aggregate metrics per model
                statistical_tests TEXT, -- JSON with p-values, confidence intervals
                best_model_id TEXT,
                comparison_summary TEXT,
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (best_model_id) REFERENCES models (model_id)
            )
        """)

        # 14. Explainability Artifacts table - Model explanation storage
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS explainability_artifacts (
                explanation_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_type TEXT CHECK(model_type IN ('core15+', 'cnn_spectrogram', 'rnn_temporal', 'transformer', 'ensemble')),
                explanation_type TEXT CHECK(explanation_type IN ('grad_cam', 'attention', 'feature_attribution', 'ensemble', 'integrated_gradients', 'shap')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                data_hash TEXT NOT NULL,
                model_version TEXT,
                confidence_score REAL,
                predicted_class INTEGER,
                explanation_config TEXT, -- JSON
                quality_metrics TEXT, -- JSON
                file_paths TEXT, -- JSON
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)

        # 15. Explanation Data Storage table - Complex nested explanation data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS explanation_data (
                explanation_id TEXT,
                data_key TEXT,
                data_value BLOB,
                data_type TEXT, -- "json", "numpy", "pickle"
                compression TEXT, -- "none", "gzip"
                PRIMARY KEY (explanation_id, data_key),
                FOREIGN KEY (explanation_id) REFERENCES explainability_artifacts (explanation_id)
            )
        """)

        # 16. Explanation Visualizations table - Visualization storage
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS explanation_visualizations (
                explanation_id TEXT,
                viz_name TEXT,
                viz_type TEXT, -- "plot", "heatmap", "attention_map", "feature_plot"
                viz_data BLOB, -- PNG/SVG bytes
                viz_metadata TEXT, -- JSON
                PRIMARY KEY (explanation_id, viz_name),
                FOREIGN KEY (explanation_id) REFERENCES explainability_artifacts (explanation_id)
            )
        """)

        # 17. Explanation Audit Events table - Explainability-specific audit trail
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS explanation_audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT NOT NULL,
                explanation_id TEXT,
                model_id TEXT,
                subject_id TEXT,
                details TEXT, -- JSON
                severity TEXT CHECK(severity IN ('info', 'warning', 'error', 'critical')),
                ip_address TEXT,
                session_id TEXT
            )
        """)

        # 18. Explanation Safety Violations table - Track safety threshold violations
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS explanation_safety_violations (
                violation_id TEXT PRIMARY KEY,
                explanation_id TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                violation_message TEXT NOT NULL,
                severity TEXT CHECK(severity IN ('low', 'medium', 'high', 'critical')),
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                resolution_notes TEXT,
                FOREIGN KEY (explanation_id) REFERENCES explainability_artifacts (explanation_id)
            )
        """)

        # 19. Explanation Integrity Checks table - Data integrity monitoring
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS explanation_integrity_checks (
                check_id TEXT PRIMARY KEY,
                explanation_id TEXT NOT NULL,
                check_type TEXT NOT NULL, -- 'hash_verification', 'format_validation', 'completeness_check'
                check_status TEXT CHECK(check_status IN ('passed', 'failed', 'warning')),
                check_details TEXT, -- JSON
                performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (explanation_id) REFERENCES explainability_artifacts (explanation_id)
            )
        """)

        # 20. Explanation Drift Monitoring table - Model explanation drift detection
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS explanation_drift_monitoring (
                drift_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                baseline_period_start TIMESTAMP,
                baseline_period_end TIMESTAMP,
                current_period_start TIMESTAMP,
                current_period_end TIMESTAMP,
                drift_score REAL,
                drift_threshold REAL,
                drift_detected BOOLEAN,
                drift_type TEXT, -- 'confidence_drift', 'attribution_drift', 'attention_drift'
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)

        # 21. Stress Test Results table - Phase VIII robustness testing
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stress_test_results (
                test_id TEXT PRIMARY KEY,
                test_type TEXT NOT NULL,
                perturbation_params TEXT, -- JSON
                subject_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                baseline_metrics TEXT, -- JSON
                stressed_metrics TEXT, -- JSON
                performance_drop TEXT, -- JSON
                passed_thresholds BOOLEAN,
                execution_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                explanation_available BOOLEAN DEFAULT FALSE
            )
        """)

        # 22. Stress Test Campaigns table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stress_test_campaigns (
                campaign_id TEXT PRIMARY KEY,
                campaign_name TEXT NOT NULL,
                config_params TEXT, -- JSON
                total_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                overall_pass_rate REAL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled'))
            )
        """)

        # 23. Cross-device validation results
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_device_results (
                test_id TEXT PRIMARY KEY,
                source_device TEXT NOT NULL,
                target_device TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                original_metrics TEXT, -- JSON
                adapted_metrics TEXT, -- JSON
                performance_degradation TEXT, -- JSON
                adaptation_method TEXT,
                adaptation_successful BOOLEAN,
                execution_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 24. Device profiles and compatibility
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS device_profiles (
                device_id TEXT PRIMARY KEY,
                manufacturer TEXT NOT NULL,
                model_name TEXT NOT NULL,
                specifications TEXT, -- JSON
                signature_characteristics TEXT, -- JSON
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS device_compatibility (
                compatibility_id TEXT PRIMARY KEY,
                device_a TEXT NOT NULL,
                device_b TEXT NOT NULL,
                compatibility_score REAL,
                adaptation_difficulty TEXT, -- 'easy', 'medium', 'hard', 'impossible'
                recommended_method TEXT,
                validation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (device_a) REFERENCES device_profiles (device_id),
                FOREIGN KEY (device_b) REFERENCES device_profiles (device_id)
            )
        """)

        # 25. Robustness audit events
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS robustness_audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT CHECK(event_type IN (
                    'stress_test_campaign_start', 'stress_test_campaign_complete',
                    'stress_test_individual', 'cross_device_validation',
                    'statistical_analysis', 'confidence_calculation',
                    'robustness_threshold_violation', 'performance_degradation_alert'
                )),
                campaign_id TEXT,
                test_id TEXT,
                model_id TEXT,
                subject_id TEXT,
                test_conditions TEXT, -- JSON
                results_summary TEXT, -- JSON
                risk_assessment TEXT, -- JSON
                compliance_status TEXT CHECK(compliance_status IN ('compliant', 'warning', 'violation')),
                user_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT -- JSON
            )
        """)

        # 26. Performance thresholds
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_thresholds (
                threshold_id TEXT PRIMARY KEY,
                threshold_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                threshold_type TEXT CHECK(threshold_type IN ('minimum', 'maximum', 'range')),
                threshold_value REAL,
                threshold_range_min REAL,
                threshold_range_max REAL,
                severity TEXT CHECK(severity IN ('info', 'warning', 'critical')),
                regulatory_basis TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active BOOLEAN DEFAULT TRUE
            )
        """)

        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_trial_site ON jobs (trial_id, site_id)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_submitted ON jobs (submitted_at)",
            "CREATE INDEX IF NOT EXISTS idx_results_job ON results (job_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_events (entity_type, entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_subjects_trial_site ON subjects (trial_id, site_id)",
            "CREATE INDEX IF NOT EXISTS idx_features_job ON feature_metadata (job_id)",
            "CREATE INDEX IF NOT EXISTS idx_features_type ON feature_metadata (feature_type, feature_name)",
            # Deep learning indexes
            "CREATE INDEX IF NOT EXISTS idx_models_type_status ON models (model_type, status)",
            "CREATE INDEX IF NOT EXISTS idx_models_deployed ON models (is_deployed, status)",
            "CREATE INDEX IF NOT EXISTS idx_training_runs_model ON training_runs (model_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_training_runs_time ON training_runs (started_at, completed_at)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_job_model ON model_predictions (job_id, model_id)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_subject ON model_predictions (subject_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_comparisons_experiment ON model_comparisons (experiment_name, created_at)",
            # Explainability indexes
            "CREATE INDEX IF NOT EXISTS idx_artifacts_subject ON explainability_artifacts (subject_id)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_model ON explainability_artifacts (model_id)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_type ON explainability_artifacts (explanation_type)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_created ON explainability_artifacts (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_explanation_audit_timestamp ON explanation_audit_events (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_explanation_audit_user ON explanation_audit_events (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_explanation_audit_explanation ON explanation_audit_events (explanation_id)",
            "CREATE INDEX IF NOT EXISTS idx_violations_explanation ON explanation_safety_violations (explanation_id)",
            "CREATE INDEX IF NOT EXISTS idx_violations_severity ON explanation_safety_violations (severity)",
            "CREATE INDEX IF NOT EXISTS idx_integrity_explanation ON explanation_integrity_checks (explanation_id)",
            "CREATE INDEX IF NOT EXISTS idx_drift_model ON explanation_drift_monitoring (model_id, detected_at)",
            # Robustness testing indexes
            "CREATE INDEX IF NOT EXISTS idx_stress_results_model ON stress_test_results (model_id)",
            "CREATE INDEX IF NOT EXISTS idx_stress_results_type ON stress_test_results (test_type)",
            "CREATE INDEX IF NOT EXISTS idx_stress_results_passed ON stress_test_results (passed_thresholds)",
            "CREATE INDEX IF NOT EXISTS idx_stress_campaigns_status ON stress_test_campaigns (status)",
            "CREATE INDEX IF NOT EXISTS idx_cross_device_source ON cross_device_results (source_device)",
            "CREATE INDEX IF NOT EXISTS idx_cross_device_target ON cross_device_results (target_device)",
            "CREATE INDEX IF NOT EXISTS idx_cross_device_model ON cross_device_results (model_id)",
            "CREATE INDEX IF NOT EXISTS idx_compatibility_devices ON device_compatibility (device_a, device_b)",
            "CREATE INDEX IF NOT EXISTS idx_robustness_audit_campaign ON robustness_audit_events (campaign_id)",
            "CREATE INDEX IF NOT EXISTS idx_robustness_audit_model ON robustness_audit_events (model_id)",
            "CREATE INDEX IF NOT EXISTS idx_thresholds_active ON performance_thresholds (active, metric_name)"
        ]

        for index_sql in indexes:
            self.conn.execute(index_sql)

        self.conn.commit()
        print("✅ Database schema created successfully")

    def insert_default_data(self):
        """Insert default configuration and test data."""

        # Default system configuration
        default_configs = [
            ('proc_001', 'processing', 'default_feature_set', 'Core15+', 'string', 'Default feature set for analysis', False),
            ('proc_002', 'processing', 'max_concurrent_jobs', '4', 'integer', 'Maximum concurrent processing jobs', False),
            ('proc_003', 'processing', 'job_timeout_minutes', '30', 'integer', 'Job timeout in minutes', False),
            ('qc_001', 'quality', 'min_signal_quality', 'fair', 'string', 'Minimum acceptable signal quality', False),
            ('qc_002', 'quality', 'max_artifacts_percent', '20', 'integer', 'Maximum artifacts percentage', False),
            ('qc_003', 'quality', 'min_recording_duration', '120', 'integer', 'Minimum recording duration in seconds', False),
            ('sec_001', 'security', 'session_timeout_hours', '8', 'integer', 'User session timeout', False),
            ('sec_002', 'security', 'max_failed_logins', '3', 'integer', 'Maximum failed login attempts', False),
            ('notif_001', 'notification', 'email_alerts_enabled', 'true', 'boolean', 'Enable email notifications', False),
            ('notif_002', 'notification', 'admin_email', 'admin@example.com', 'string', 'Administrator email', True),
            # Deep learning configurations
            ('dl_001', 'deep_learning', 'default_batch_size', '32', 'integer', 'Default batch size for training', False),
            ('dl_002', 'deep_learning', 'max_epochs', '100', 'integer', 'Maximum training epochs', False),
            ('dl_003', 'deep_learning', 'learning_rate', '0.001', 'float', 'Default learning rate', False),
            ('dl_004', 'deep_learning', 'use_gpu', 'true', 'boolean', 'Enable GPU acceleration', False),
            ('dl_005', 'deep_learning', 'model_checkpoint_dir', 'models/checkpoints', 'string', 'Model checkpoint directory', False),
            ('dl_006', 'deep_learning', 'early_stopping_patience', '10', 'integer', 'Early stopping patience', False),
            # Explainability configurations
            ('exp_001', 'explainability', 'grad_cam_target_layer', 'conv_layers.4', 'string', 'Default Grad-CAM target layer', False),
            ('exp_002', 'explainability', 'attention_rollout_enabled', 'true', 'boolean', 'Enable attention rollout', False),
            ('exp_003', 'explainability', 'shap_background_samples', '100', 'integer', 'SHAP background sample size', False),
            ('exp_004', 'explainability', 'min_confidence_threshold', '0.5', 'float', 'Minimum confidence for explanations', False),
            ('exp_005', 'explainability', 'explanation_timeout', '300', 'integer', 'Explanation generation timeout (seconds)', False),
            ('exp_006', 'explainability', 'safety_checks_enabled', 'true', 'boolean', 'Enable safety validation', False)
        ]

        for config in default_configs:
            self.conn.execute("""
                INSERT OR REPLACE INTO system_config
                (config_id, config_category, config_key, config_value, config_type, description, is_sensitive)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, config)

        # Example trial for testing
        self.conn.execute("""
            INSERT OR REPLACE INTO trials
            (trial_id, trial_name, protocol_version, irb_number, pi_name, institution, status, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'TRIAL_001',
            'Phase IV EEG Biomarker Validation',
            '1.0',
            'IRB-2025-001',
            'Dr. John Smith',
            'Academic Medical Center',
            'planning',
            'Multi-center prospective validation of Core15+ EEG biomarkers'
        ))

        # Example sites
        sites_data = [
            ('SITE_001', 'Academic Medical Center', 'University Hospital', 'USA', 'Dr. John Smith', 'coord1@example.com', 'active'),
            ('SITE_002', 'Movement Disorders Clinic', 'Specialty Center', 'USA', 'Dr. Jane Doe', 'coord2@example.com', 'active'),
            ('SITE_003', 'International Site', 'European Hospital', 'EU', 'Dr. Hans Mueller', 'coord3@example.com', 'pending')
        ]

        for site in sites_data:
            self.conn.execute("""
                INSERT OR REPLACE INTO sites
                (site_id, site_name, institution, country, pi_name, coordinator_email, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, site)

        # Link trial and sites
        for site_id in ['SITE_001', 'SITE_002', 'SITE_003']:
            self.conn.execute("""
                INSERT OR REPLACE INTO trial_sites
                (trial_id, site_id, enrollment_target, status)
                VALUES (?, ?, ?, ?)
            """, ('TRIAL_001', site_id, 67, 'pending'))

        self.conn.commit()
        print("✅ Default data inserted successfully")

    def create_views(self):
        """Create useful views for reporting and monitoring."""

        # Job summary view
        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS job_summary AS
            SELECT
                j.job_id,
                j.trial_id,
                j.site_id,
                j.subject_id,
                j.job_type,
                j.feature_set,
                j.status,
                j.submitted_at,
                j.completed_at,
                j.duration_seconds,
                r.prediction,
                r.confidence_score,
                r.qc_pass,
                r.signal_quality,
                s.site_name,
                t.trial_name
            FROM jobs j
            LEFT JOIN results r ON j.job_id = r.job_id
            LEFT JOIN sites s ON j.site_id = s.site_id
            LEFT JOIN trials t ON j.trial_id = t.trial_id
        """)

        # Trial enrollment view
        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS trial_enrollment AS
            SELECT
                t.trial_id,
                t.trial_name,
                ts.site_id,
                s.site_name,
                ts.enrollment_target,
                ts.enrollment_actual,
                COUNT(subj.subject_id) as subjects_enrolled,
                COUNT(CASE WHEN subj.status = 'completed' THEN 1 END) as subjects_completed
            FROM trials t
            JOIN trial_sites ts ON t.trial_id = ts.trial_id
            JOIN sites s ON ts.site_id = s.site_id
            LEFT JOIN subjects subj ON t.trial_id = subj.trial_id AND ts.site_id = subj.site_id
            GROUP BY t.trial_id, ts.site_id
        """)

        # QC metrics view
        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS qc_metrics AS
            SELECT
                j.trial_id,
                j.site_id,
                j.feature_set,
                COUNT(*) as total_jobs,
                COUNT(CASE WHEN r.qc_pass = 1 THEN 1 END) as qc_passed,
                COUNT(CASE WHEN r.signal_quality = 'excellent' THEN 1 END) as excellent_quality,
                COUNT(CASE WHEN r.signal_quality = 'good' THEN 1 END) as good_quality,
                COUNT(CASE WHEN r.signal_quality = 'fair' THEN 1 END) as fair_quality,
                COUNT(CASE WHEN r.signal_quality = 'poor' THEN 1 END) as poor_quality,
                AVG(r.qc_score) as avg_qc_score,
                AVG(r.balanced_accuracy) as avg_balanced_accuracy
            FROM jobs j
            JOIN results r ON j.job_id = r.job_id
            WHERE j.status = 'completed'
            GROUP BY j.trial_id, j.site_id, j.feature_set
        """)

        self.conn.commit()
        print("✅ Database views created successfully")

    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Create database schema for EEG clinical platform."""

    print("="*60)
    print("EEG PLATFORM DATABASE SCHEMA CREATION")
    print("Phase VI: Clinical-Grade Persistence")
    print("="*60)

    # Create database
    db = DatabaseManager()

    try:
        # Create schema
        print("\n📋 Creating database schema...")
        db.create_schema()

        # Insert default data
        print("\n📝 Inserting default configuration...")
        db.insert_default_data()

        # Create views
        print("\n👁️  Creating database views...")
        db.create_views()

        # Summary
        print("\n📊 Database Summary:")
        cursor = db.conn.cursor()

        # Count tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"  Tables created: {len(tables)}")

        # Count views
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = cursor.fetchall()
        print(f"  Views created: {len(views)}")

        # Count indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
        indexes = cursor.fetchall()
        print(f"  Indexes created: {len(indexes)}")

        # Show configuration
        cursor.execute("SELECT COUNT(*) FROM system_config")
        config_count = cursor.fetchone()[0]
        print(f"  Configuration entries: {config_count}")

        print(f"\n✅ Database created successfully: {db.db_path}")
        print("\n🎯 Ready for clinical platform deployment!")

    except Exception as e:
        print(f"\n❌ Error creating database: {e}")
        return 1

    finally:
        db.close()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())