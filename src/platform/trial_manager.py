#!/usr/bin/env python3
"""
Clinical Trial Management System
Phase VI-B: Comprehensive trial metadata and regulatory compliance
"""

import sqlite3
import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from enum import Enum

class TrialStatus(Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    TERMINATED = "terminated"

class SubjectStatus(Enum):
    SCREENED = "screened"
    ENROLLED = "enrolled"
    COMPLETED = "completed"
    WITHDRAWN = "withdrawn"
    EXCLUDED = "excluded"

@dataclass
class TrialMetadata:
    """Comprehensive trial metadata for regulatory compliance."""
    trial_id: str
    trial_name: str
    protocol_version: str
    irb_number: str
    pi_name: str
    institution: str
    start_date: Optional[date]
    end_date: Optional[date]
    status: TrialStatus
    description: str
    primary_endpoint: str
    secondary_endpoints: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    sample_size_target: int
    statistical_plan: Dict
    regulatory_approvals: Dict
    created_at: datetime
    updated_at: datetime

@dataclass
class SiteMetadata:
    """Clinical site metadata and certification tracking."""
    site_id: str
    site_name: str
    institution: str
    country: str
    pi_name: str
    coordinator_email: str
    status: str
    certification_date: Optional[date]
    last_audit_date: Optional[date]
    regulatory_documents: Dict
    training_records: List[Dict]
    contact_info: Dict

@dataclass
class SubjectMetadata:
    """De-identified subject metadata for clinical tracking."""
    subject_id: str
    trial_id: str
    site_id: str
    screening_number: str
    enrollment_date: Optional[date]
    group_assignment: str
    age_group: str
    gender: str
    status: SubjectStatus
    visit_schedule: List[Dict]
    protocol_deviations: List[Dict]
    adverse_events: List[Dict]
    completion_date: Optional[date]
    withdrawal_reason: Optional[str]

class ClinicalTrialManager:
    """Comprehensive trial management with regulatory compliance."""

    def __init__(self, db_path: str = "data/eeg_platform.db"):
        """Initialize trial manager with database connection."""
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)

        # Ensure database exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def get_db_connection(self):
        """Get database connection with proper configuration."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def audit_log(self, event_type: str, entity_type: str, entity_id: str,
                  action: str, details: Dict = None, user_id: str = "system"):
        """Log audit event for regulatory compliance."""
        conn = self.get_db_connection()
        try:
            conn.execute("""
                INSERT INTO audit_events
                (event_type, entity_type, entity_id, user_id, action, details, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event_type, entity_type, entity_id, user_id, action,
                json.dumps(details, default=str) if details else None, True
            ))
            conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
        finally:
            conn.close()

    # ================================================================================
    # TRIAL MANAGEMENT
    # ================================================================================

    def create_comprehensive_trial(self, trial_metadata: TrialMetadata, user_id: str = "system") -> str:
        """Create comprehensive trial with full metadata."""
        conn = self.get_db_connection()
        try:
            # Insert main trial record
            conn.execute("""
                INSERT INTO trials
                (trial_id, trial_name, protocol_version, irb_number, pi_name,
                 institution, start_date, end_date, status, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trial_metadata.trial_id,
                trial_metadata.trial_name,
                trial_metadata.protocol_version,
                trial_metadata.irb_number,
                trial_metadata.pi_name,
                trial_metadata.institution,
                trial_metadata.start_date.isoformat() if trial_metadata.start_date else None,
                trial_metadata.end_date.isoformat() if trial_metadata.end_date else None,
                trial_metadata.status.value,
                trial_metadata.description
            ))

            # Store extended metadata in system_config
            extended_config = {
                'primary_endpoint': trial_metadata.primary_endpoint,
                'secondary_endpoints': trial_metadata.secondary_endpoints,
                'inclusion_criteria': trial_metadata.inclusion_criteria,
                'exclusion_criteria': trial_metadata.exclusion_criteria,
                'sample_size_target': trial_metadata.sample_size_target,
                'statistical_plan': trial_metadata.statistical_plan,
                'regulatory_approvals': trial_metadata.regulatory_approvals
            }

            conn.execute("""
                INSERT INTO system_config
                (config_id, config_category, config_key, config_value, config_type,
                 description, is_sensitive)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"trial_{trial_metadata.trial_id}_metadata",
                "trial_metadata",
                f"{trial_metadata.trial_id}_extended",
                json.dumps(extended_config, default=str),
                "json",
                f"Extended metadata for trial {trial_metadata.trial_id}",
                False
            ))

            conn.commit()

            self.audit_log('trial_create', 'trial', trial_metadata.trial_id,
                          'create_comprehensive_trial',
                          {'trial_name': trial_metadata.trial_name,
                           'institution': trial_metadata.institution,
                           'sample_size': trial_metadata.sample_size_target}, user_id)

            self.logger.info(f"Created comprehensive trial {trial_metadata.trial_id}")
            return trial_metadata.trial_id

        except Exception as e:
            self.logger.error(f"Failed to create trial: {e}")
            raise
        finally:
            conn.close()

    def update_trial_status(self, trial_id: str, new_status: TrialStatus,
                           reason: str = None, user_id: str = "system"):
        """Update trial status with audit trail."""
        conn = self.get_db_connection()
        try:
            # Get current status
            cursor = conn.execute("SELECT status FROM trials WHERE trial_id = ?", (trial_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Trial not found: {trial_id}")

            old_status = row[0]

            # Update status
            conn.execute("""
                UPDATE trials
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE trial_id = ?
            """, (new_status.value, trial_id))

            conn.commit()

            self.audit_log('trial_status_change', 'trial', trial_id,
                          f'status_change_{old_status}_to_{new_status.value}',
                          {'old_status': old_status, 'new_status': new_status.value,
                           'reason': reason}, user_id)

            self.logger.info(f"Updated trial {trial_id} status: {old_status} → {new_status.value}")

        except Exception as e:
            self.logger.error(f"Failed to update trial status: {e}")
            raise
        finally:
            conn.close()

    def get_trial_enrollment_summary(self, trial_id: str) -> Dict:
        """Get comprehensive enrollment summary for trial."""
        conn = self.get_db_connection()
        try:
            # Overall enrollment
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_enrolled,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'withdrawn' THEN 1 END) as withdrawn,
                    COUNT(CASE WHEN group_assignment = 'PD' THEN 1 END) as pd_subjects,
                    COUNT(CASE WHEN group_assignment = 'CONTROL' THEN 1 END) as control_subjects
                FROM subjects
                WHERE trial_id = ?
            """, (trial_id,))
            overall = dict(cursor.fetchone())

            # Site-wise enrollment
            cursor = conn.execute("""
                SELECT
                    s.site_name,
                    s.site_id,
                    ts.enrollment_target,
                    COUNT(subj.subject_id) as enrolled,
                    COUNT(CASE WHEN subj.status = 'completed' THEN 1 END) as completed
                FROM trial_sites ts
                JOIN sites s ON ts.site_id = s.site_id
                LEFT JOIN subjects subj ON ts.trial_id = subj.trial_id AND ts.site_id = subj.site_id
                WHERE ts.trial_id = ?
                GROUP BY s.site_id, s.site_name, ts.enrollment_target
            """, (trial_id,))
            sites = [dict(row) for row in cursor.fetchall()]

            # Enrollment timeline
            cursor = conn.execute("""
                SELECT
                    DATE(enrollment_date) as date,
                    COUNT(*) as enrollments
                FROM subjects
                WHERE trial_id = ? AND enrollment_date IS NOT NULL
                GROUP BY DATE(enrollment_date)
                ORDER BY enrollment_date
            """, (trial_id,))
            timeline = [dict(row) for row in cursor.fetchall()]

            return {
                'trial_id': trial_id,
                'overall_statistics': overall,
                'site_breakdown': sites,
                'enrollment_timeline': timeline,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get enrollment summary: {e}")
            raise
        finally:
            conn.close()

    # ================================================================================
    # SITE MANAGEMENT
    # ================================================================================

    def register_comprehensive_site(self, site_metadata: SiteMetadata, user_id: str = "system") -> str:
        """Register site with comprehensive metadata."""
        conn = self.get_db_connection()
        try:
            # Insert main site record
            conn.execute("""
                INSERT INTO sites
                (site_id, site_name, institution, country, pi_name,
                 coordinator_email, status, certification_date, last_audit_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                site_metadata.site_id,
                site_metadata.site_name,
                site_metadata.institution,
                site_metadata.country,
                site_metadata.pi_name,
                site_metadata.coordinator_email,
                site_metadata.status,
                site_metadata.certification_date.isoformat() if site_metadata.certification_date else None,
                site_metadata.last_audit_date.isoformat() if site_metadata.last_audit_date else None
            ))

            # Store extended metadata
            extended_config = {
                'regulatory_documents': site_metadata.regulatory_documents,
                'training_records': site_metadata.training_records,
                'contact_info': site_metadata.contact_info
            }

            conn.execute("""
                INSERT INTO system_config
                (config_id, config_category, config_key, config_value, config_type,
                 description, is_sensitive)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"site_{site_metadata.site_id}_metadata",
                "site_metadata",
                f"{site_metadata.site_id}_extended",
                json.dumps(extended_config, default=str),
                "json",
                f"Extended metadata for site {site_metadata.site_id}",
                True  # Site info is sensitive
            ))

            conn.commit()

            self.audit_log('site_register', 'site', site_metadata.site_id,
                          'register_comprehensive_site',
                          {'site_name': site_metadata.site_name,
                           'institution': site_metadata.institution,
                           'country': site_metadata.country}, user_id)

            self.logger.info(f"Registered comprehensive site {site_metadata.site_id}")
            return site_metadata.site_id

        except Exception as e:
            self.logger.error(f"Failed to register site: {e}")
            raise
        finally:
            conn.close()

    def update_site_certification(self, site_id: str, certification_date: date,
                                 documents: Dict, user_id: str = "system"):
        """Update site certification status."""
        conn = self.get_db_connection()
        try:
            conn.execute("""
                UPDATE sites
                SET certification_date = ?, status = 'active', updated_at = CURRENT_TIMESTAMP
                WHERE site_id = ?
            """, (certification_date.isoformat(), site_id))

            # Update certification documents
            conn.execute("""
                UPDATE system_config
                SET config_value = json_patch(config_value, ?)
                WHERE config_id = ?
            """, (
                json.dumps({'regulatory_documents': documents}, default=str),
                f"site_{site_id}_metadata"
            ))

            conn.commit()

            self.audit_log('site_certification', 'site', site_id,
                          'update_certification',
                          {'certification_date': certification_date.isoformat(),
                           'documents': list(documents.keys())}, user_id)

            self.logger.info(f"Updated certification for site {site_id}")

        except Exception as e:
            self.logger.error(f"Failed to update site certification: {e}")
            raise
        finally:
            conn.close()

    # ================================================================================
    # SUBJECT MANAGEMENT
    # ================================================================================

    def enroll_comprehensive_subject(self, subject_metadata: SubjectMetadata,
                                   user_id: str = "system") -> str:
        """Enroll subject with comprehensive metadata."""
        conn = self.get_db_connection()
        try:
            # Insert main subject record
            conn.execute("""
                INSERT INTO subjects
                (subject_id, trial_id, site_id, screening_number, enrollment_date,
                 group_assignment, age_group, gender, status, withdrawal_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                subject_metadata.subject_id,
                subject_metadata.trial_id,
                subject_metadata.site_id,
                subject_metadata.screening_number,
                subject_metadata.enrollment_date.isoformat() if subject_metadata.enrollment_date else None,
                subject_metadata.group_assignment,
                subject_metadata.age_group,
                subject_metadata.gender,
                subject_metadata.status.value,
                subject_metadata.withdrawal_reason
            ))

            # Store extended metadata
            extended_config = {
                'visit_schedule': subject_metadata.visit_schedule,
                'protocol_deviations': subject_metadata.protocol_deviations,
                'adverse_events': subject_metadata.adverse_events,
                'completion_date': subject_metadata.completion_date.isoformat() if subject_metadata.completion_date else None
            }

            conn.execute("""
                INSERT INTO system_config
                (config_id, config_category, config_key, config_value, config_type,
                 description, is_sensitive)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"subject_{subject_metadata.subject_id}_metadata",
                "subject_metadata",
                f"{subject_metadata.subject_id}_extended",
                json.dumps(extended_config, default=str),
                "json",
                f"Extended metadata for subject {subject_metadata.subject_id}",
                True  # Subject data is sensitive
            ))

            conn.commit()

            # Update site enrollment count
            conn.execute("""
                UPDATE trial_sites
                SET enrollment_actual = enrollment_actual + 1
                WHERE trial_id = ? AND site_id = ?
            """, (subject_metadata.trial_id, subject_metadata.site_id))

            conn.commit()

            self.audit_log('subject_enroll', 'subject', subject_metadata.subject_id,
                          'enroll_comprehensive_subject',
                          {'trial_id': subject_metadata.trial_id,
                           'site_id': subject_metadata.site_id,
                           'group': subject_metadata.group_assignment}, user_id)

            self.logger.info(f"Enrolled comprehensive subject {subject_metadata.subject_id}")
            return subject_metadata.subject_id

        except Exception as e:
            self.logger.error(f"Failed to enroll subject: {e}")
            raise
        finally:
            conn.close()

    def record_protocol_deviation(self, subject_id: str, deviation_type: str,
                                description: str, severity: str, corrective_action: str,
                                user_id: str = "system"):
        """Record protocol deviation for subject."""
        deviation_id = f"DEV_{uuid.uuid4().hex[:8].upper()}"

        deviation_record = {
            'deviation_id': deviation_id,
            'type': deviation_type,
            'description': description,
            'severity': severity,
            'corrective_action': corrective_action,
            'date_reported': datetime.now().isoformat(),
            'reported_by': user_id
        }

        conn = self.get_db_connection()
        try:
            # Get current deviations
            cursor = conn.execute("""
                SELECT config_value FROM system_config
                WHERE config_id = ?
            """, (f"subject_{subject_id}_metadata",))
            row = cursor.fetchone()

            if row:
                metadata = json.loads(row[0])
                if 'protocol_deviations' not in metadata:
                    metadata['protocol_deviations'] = []
                metadata['protocol_deviations'].append(deviation_record)

                conn.execute("""
                    UPDATE system_config
                    SET config_value = ?
                    WHERE config_id = ?
                """, (
                    json.dumps(metadata, default=str),
                    f"subject_{subject_id}_metadata"
                ))

                conn.commit()

                self.audit_log('protocol_deviation', 'subject', subject_id,
                              'record_deviation',
                              {'deviation_id': deviation_id, 'type': deviation_type,
                               'severity': severity}, user_id)

                self.logger.info(f"Recorded protocol deviation {deviation_id} for {subject_id}")

        except Exception as e:
            self.logger.error(f"Failed to record protocol deviation: {e}")
            raise
        finally:
            conn.close()

    # ================================================================================
    # REGULATORY REPORTING
    # ================================================================================

    def generate_regulatory_report(self, trial_id: str, report_type: str = "enrollment") -> Dict:
        """Generate regulatory compliance report."""
        conn = self.get_db_connection()
        try:
            if report_type == "enrollment":
                return self._generate_enrollment_report(conn, trial_id)
            elif report_type == "safety":
                return self._generate_safety_report(conn, trial_id)
            elif report_type == "quality":
                return self._generate_quality_report(conn, trial_id)
            elif report_type == "audit_trail":
                return self._generate_audit_trail_report(conn, trial_id)
            else:
                raise ValueError(f"Unknown report type: {report_type}")

        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            raise
        finally:
            conn.close()

    def _generate_enrollment_report(self, conn: sqlite3.Connection, trial_id: str) -> Dict:
        """Generate enrollment report for regulators."""
        # Trial overview
        cursor = conn.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,))
        trial_info = dict(cursor.fetchone())

        # Enrollment statistics
        enrollment_summary = self.get_trial_enrollment_summary(trial_id)

        # Protocol deviations summary
        cursor = conn.execute("""
            SELECT COUNT(*) as total_subjects,
                   COUNT(CASE WHEN config_value LIKE '%protocol_deviations%' THEN 1 END) as subjects_with_deviations
            FROM system_config sc
            JOIN subjects s ON sc.config_key = s.subject_id || '_extended'
            WHERE s.trial_id = ? AND sc.config_category = 'subject_metadata'
        """, (trial_id,))
        deviation_stats = dict(cursor.fetchone())

        return {
            'report_type': 'enrollment',
            'trial_id': trial_id,
            'trial_info': trial_info,
            'enrollment_summary': enrollment_summary,
            'protocol_deviation_summary': deviation_stats,
            'generated_at': datetime.now().isoformat(),
            'generated_for': 'regulatory_submission'
        }

    def _generate_safety_report(self, conn: sqlite3.Connection, trial_id: str) -> Dict:
        """Generate safety report for DSMB."""
        # Adverse events (would be stored in subject metadata)
        cursor = conn.execute("""
            SELECT s.subject_id, s.group_assignment, sc.config_value
            FROM subjects s
            JOIN system_config sc ON sc.config_key = s.subject_id || '_extended'
            WHERE s.trial_id = ? AND sc.config_category = 'subject_metadata'
        """, (trial_id,))

        ae_summary = {'total_aes': 0, 'serious_aes': 0, 'by_group': {}}
        for row in cursor.fetchall():
            metadata = json.loads(row[2])
            aes = metadata.get('adverse_events', [])
            group = row[1]

            if group not in ae_summary['by_group']:
                ae_summary['by_group'][group] = {'total': 0, 'serious': 0}

            for ae in aes:
                ae_summary['total_aes'] += 1
                ae_summary['by_group'][group]['total'] += 1
                if ae.get('severity') == 'serious':
                    ae_summary['serious_aes'] += 1
                    ae_summary['by_group'][group]['serious'] += 1

        return {
            'report_type': 'safety',
            'trial_id': trial_id,
            'adverse_event_summary': ae_summary,
            'generated_at': datetime.now().isoformat(),
            'generated_for': 'dsmb_review'
        }

    def _generate_audit_trail_report(self, conn: sqlite3.Connection, trial_id: str) -> Dict:
        """Generate audit trail report for compliance."""
        cursor = conn.execute("""
            SELECT event_type, action, COUNT(*) as count
            FROM audit_events
            WHERE entity_id LIKE ? OR details LIKE ?
            GROUP BY event_type, action
            ORDER BY count DESC
        """, (f"%{trial_id}%", f"%{trial_id}%"))

        audit_summary = [dict(row) for row in cursor.fetchall()]

        # Recent significant events
        cursor = conn.execute("""
            SELECT *
            FROM audit_events
            WHERE (entity_id LIKE ? OR details LIKE ?)
              AND event_type IN ('trial_create', 'trial_status_change', 'subject_enroll',
                                'protocol_deviation', 'job_submit', 'result_save')
            ORDER BY timestamp DESC
            LIMIT 100
        """, (f"%{trial_id}%", f"%{trial_id}%"))

        recent_events = [dict(row) for row in cursor.fetchall()]

        return {
            'report_type': 'audit_trail',
            'trial_id': trial_id,
            'audit_summary': audit_summary,
            'recent_significant_events': recent_events,
            'generated_at': datetime.now().isoformat(),
            'generated_for': 'regulatory_audit'
        }

def main():
    """Demo comprehensive trial management."""
    print("="*60)
    print("CLINICAL TRIAL MANAGEMENT SYSTEM")
    print("Phase VI-B: Comprehensive Metadata & Regulatory Compliance")
    print("="*60)

    manager = ClinicalTrialManager()

    # Create comprehensive trial
    trial_metadata = TrialMetadata(
        trial_id="TRIAL_PHASEIV_001",
        trial_name="Phase IV EEG Biomarker Validation Study",
        protocol_version="1.0",
        irb_number="IRB-2025-001",
        pi_name="Dr. Sarah Chen",
        institution="Academic Medical Center",
        start_date=date(2025, 1, 15),
        end_date=date(2027, 1, 15),
        status=TrialStatus.PLANNING,
        description="Multi-center prospective validation of Core15+ EEG biomarkers",
        primary_endpoint="Balanced accuracy ≥87.5% for PD vs Control classification",
        secondary_endpoints=[
            "Sensitivity ≥90%",
            "Specificity ≥85%",
            "Cross-site consistency (CV < 10%)"
        ],
        inclusion_criteria=[
            "Age 45-80 years",
            "Clinical diagnosis of idiopathic PD (MDS criteria)",
            "Hoehn & Yahr stage I-III",
            "Stable dopaminergic medication ≥3 months"
        ],
        exclusion_criteria=[
            "Secondary or atypical parkinsonism",
            "Significant cognitive impairment (MoCA <24)",
            "Active seizure disorder"
        ],
        sample_size_target=200,
        statistical_plan={
            "primary_analysis": "diagnostic_accuracy",
            "alpha": 0.05,
            "power": 0.80,
            "interim_analyses": 2,
            "stopping_rules": "efficacy_and_futility"
        },
        regulatory_approvals={
            "fda_ide": "pending",
            "central_irb": "approved",
            "health_canada": "pending"
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    trial_id = manager.create_comprehensive_trial(trial_metadata)
    print(f"✅ Created comprehensive trial: {trial_id}")

    # Register comprehensive site
    site_metadata = SiteMetadata(
        site_id="SITE_AMC_001",
        site_name="Academic Medical Center",
        institution="University Hospital System",
        country="USA",
        pi_name="Dr. Michael Rodriguez",
        coordinator_email="research.coord@amc.edu",
        status="pending",
        certification_date=None,
        last_audit_date=None,
        regulatory_documents={
            "irb_approval": "IRB-2025-001-AMC.pdf",
            "cv_pi": "rodriguez_cv_2025.pdf",
            "delegation_log": "delegation_log_amc.pdf"
        },
        training_records=[
            {
                "staff_id": "TECH_001",
                "training_type": "GCP",
                "completion_date": "2024-12-15",
                "certificate": "gcp_cert_001.pdf"
            }
        ],
        contact_info={
            "phone": "+1-555-0123",
            "address": "123 Medical Center Dr, Research City, ST 12345",
            "emergency_contact": "+1-555-0124"
        }
    )

    site_id = manager.register_comprehensive_site(site_metadata)
    print(f"✅ Registered comprehensive site: {site_id}")

    # Update certification
    manager.update_site_certification(
        site_id,
        date(2025, 1, 10),
        {"certification_letter": "cert_amc_2025.pdf"}
    )
    print(f"✅ Updated site certification")

    # Enroll comprehensive subjects
    for i in range(5):
        subject_metadata = SubjectMetadata(
            subject_id=f"SUBJ_AMC_{i+1:03d}",
            trial_id=trial_id,
            site_id=site_id,
            screening_number=f"SCR-AMC-{i+1:03d}",
            enrollment_date=date(2025, 2, 1 + i),
            group_assignment="PD" if i < 3 else "CONTROL",
            age_group="46-60",
            gender="M" if i % 2 == 0 else "F",
            status=SubjectStatus.ENROLLED,
            visit_schedule=[
                {"visit": "screening", "date": "2025-01-25", "completed": True},
                {"visit": "baseline", "date": "2025-02-01", "completed": True},
                {"visit": "eeg_session", "date": "2025-02-08", "completed": False}
            ],
            protocol_deviations=[],
            adverse_events=[],
            completion_date=None,
            withdrawal_reason=None
        )

        subject_id = manager.enroll_comprehensive_subject(subject_metadata)
        print(f"✅ Enrolled subject: {subject_id}")

    # Record a protocol deviation
    manager.record_protocol_deviation(
        "SUBJ_AMC_001",
        "visit_window",
        "EEG session conducted 2 days outside window due to equipment maintenance",
        "minor",
        "Session rescheduled, no impact on data quality"
    )
    print("✅ Recorded protocol deviation")

    # Generate regulatory reports
    enrollment_report = manager.generate_regulatory_report(trial_id, "enrollment")
    print(f"✅ Generated enrollment report: {len(enrollment_report['enrollment_summary']['site_breakdown'])} sites")

    audit_report = manager.generate_regulatory_report(trial_id, "audit_trail")
    print(f"✅ Generated audit trail report: {len(audit_report['recent_significant_events'])} events")

    print("\n🎯 Trial management system ready for clinical deployment!")

if __name__ == "__main__":
    main()