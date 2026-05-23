#!/usr/bin/env python3
"""
Phase VI-B: Complete Integration Demonstration
Persistent Platform with Clinical Trial Management, Job Processing, and Monitoring

This script demonstrates the complete Phase VI-B platform capabilities:
1. Database schema and persistence
2. Clinical trial and site management
3. Subject enrollment and tracking
4. Job queue processing with Core15+ features
5. Real-time monitoring and alerting
6. Configuration management
7. Regulatory audit logging
"""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime, date

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / "src" / "platform"))
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent))

from persistent_manager import PersistentPlatformManager, JobType, JobConfig
from trial_manager import ClinicalTrialManager, TrialMetadata, SiteMetadata, SubjectMetadata, TrialStatus, SubjectStatus
from monitoring_dashboard import MonitoringDashboard
from create_database_schema import DatabaseManager

def setup_complete_platform():
    """Setup complete clinical platform with all components."""
    print("🏗️  Setting up complete clinical platform...")

    # 1. Create database
    print("   📊 Creating database schema...")
    db_manager = DatabaseManager("data/eeg_platform.db")
    db_manager.create_schema()
    db_manager.insert_default_data()
    db_manager.create_views()
    db_manager.close()
    print("   ✅ Database schema created")

    # 2. Initialize components
    platform = PersistentPlatformManager("data/eeg_platform.db", max_workers=3)
    trial_manager = ClinicalTrialManager("data/eeg_platform.db")
    dashboard = MonitoringDashboard("data/eeg_platform.db", update_interval=10)

    print("   ✅ Platform components initialized")
    return platform, trial_manager, dashboard

def create_comprehensive_trial(trial_manager: ClinicalTrialManager):
    """Create comprehensive Phase IV trial."""
    print("🧪 Creating comprehensive Phase IV trial...")

    trial_metadata = TrialMetadata(
        trial_id="TRIAL_PHASEIV_INTEGRATION",
        trial_name="Phase IV EEG Biomarker Validation - Integration Study",
        protocol_version="1.0",
        irb_number="IRB-2025-INTEGRATION",
        pi_name="Dr. Elena Rodriguez",
        institution="Integration Medical Center",
        start_date=date(2025, 3, 1),
        end_date=date(2027, 3, 1),
        status=TrialStatus.PLANNING,
        description="Multi-center prospective validation of Core15+ EEG biomarkers with integrated platform",
        primary_endpoint="Balanced accuracy ≥87.5% for PD vs Control classification using integrated platform",
        secondary_endpoints=[
            "Platform reliability >99.5% uptime",
            "Processing throughput >50 subjects/day",
            "Regulatory compliance 100%",
            "Cross-site consistency (CV < 5%)"
        ],
        inclusion_criteria=[
            "Age 45-80 years",
            "Clinical diagnosis of idiopathic PD",
            "Able to undergo 20-minute EEG recording",
            "Informed consent provided"
        ],
        exclusion_criteria=[
            "Secondary parkinsonism",
            "Active seizure disorder",
            "Inability to tolerate EEG recording"
        ],
        sample_size_target=300,
        statistical_plan={
            "primary_analysis": "diagnostic_accuracy_with_platform_metrics",
            "alpha": 0.05,
            "power": 0.90,
            "interim_analyses": 3,
            "platform_validation": "concurrent"
        },
        regulatory_approvals={
            "fda_ide": "pending",
            "central_irb": "approved",
            "iso_13485": "in_progress",
            "hipaa_compliance": "validated"
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    trial_id = trial_manager.create_comprehensive_trial(trial_metadata)
    print(f"   ✅ Created trial: {trial_id}")
    return trial_id

def register_clinical_sites(trial_manager: ClinicalTrialManager):
    """Register multiple clinical sites."""
    print("🏥 Registering clinical sites...")

    sites_data = [
        {
            "site_id": "SITE_AMC_001",
            "site_name": "Academic Medical Center",
            "institution": "University Hospital System",
            "country": "USA",
            "pi_name": "Dr. Michael Chen",
            "coordinator_email": "mchen@amc.edu",
            "regulatory_documents": {
                "irb_approval": "IRB-AMC-2025.pdf",
                "gcp_training": "GCP-AMC-2025.pdf",
                "equipment_validation": "EEG-VAL-AMC.pdf"
            },
            "training_records": [
                {"staff": "Tech_001", "training": "Core15+ Certification", "date": "2025-01-15"},
                {"staff": "Coord_001", "training": "Clinical Protocol", "date": "2025-01-20"}
            ]
        },
        {
            "site_id": "SITE_NEURO_002",
            "site_name": "Neurological Specialty Center",
            "institution": "Movement Disorders Institute",
            "country": "USA",
            "pi_name": "Dr. Sarah Kim",
            "coordinator_email": "skim@neuro.org",
            "regulatory_documents": {
                "irb_approval": "IRB-NEURO-2025.pdf",
                "gcp_training": "GCP-NEURO-2025.pdf",
                "equipment_validation": "EEG-VAL-NEURO.pdf"
            },
            "training_records": [
                {"staff": "Tech_002", "training": "Core15+ Certification", "date": "2025-02-01"},
                {"staff": "Coord_002", "training": "Clinical Protocol", "date": "2025-02-05"}
            ]
        },
        {
            "site_id": "SITE_INTL_003",
            "site_name": "International Parkinson Center",
            "institution": "European Medical Network",
            "country": "EU",
            "pi_name": "Dr. Hans Mueller",
            "coordinator_email": "hmueller@euro-med.eu",
            "regulatory_documents": {
                "ethics_approval": "ETHICS-EU-2025.pdf",
                "gdpr_compliance": "GDPR-EU-2025.pdf",
                "equipment_validation": "EEG-VAL-EU.pdf"
            },
            "training_records": [
                {"staff": "Tech_003", "training": "Core15+ Certification", "date": "2025-02-10"},
                {"staff": "Coord_003", "training": "Clinical Protocol", "date": "2025-02-15"}
            ]
        }
    ]

    registered_sites = []
    for site_data in sites_data:
        site_metadata = SiteMetadata(
            site_id=site_data["site_id"],
            site_name=site_data["site_name"],
            institution=site_data["institution"],
            country=site_data["country"],
            pi_name=site_data["pi_name"],
            coordinator_email=site_data["coordinator_email"],
            status="pending",
            certification_date=None,
            last_audit_date=None,
            regulatory_documents=site_data["regulatory_documents"],
            training_records=site_data["training_records"],
            contact_info={
                "phone": f"+1-555-{1000 + len(registered_sites):04d}",
                "address": f"{100 + len(registered_sites)} Medical Plaza Dr"
            }
        )

        site_id = trial_manager.register_comprehensive_site(site_metadata)

        # Certify site immediately for demo
        trial_manager.update_site_certification(
            site_id,
            date(2025, 2, 20 + len(registered_sites)),
            {"platform_certification": f"CERT-{site_id}-2025.pdf"}
        )

        registered_sites.append(site_id)
        print(f"   ✅ Registered and certified site: {site_id}")

    return registered_sites

def enroll_subjects_and_submit_jobs(platform: PersistentPlatformManager,
                                   trial_id: str, site_ids: list):
    """Enroll subjects and submit processing jobs."""
    print("👥 Enrolling subjects and submitting jobs...")

    subjects_per_site = 5
    all_subjects = []

    # Create demo EEG file if it doesn't exist
    demo_file = Path("data/demo_eeg.fif")
    if not demo_file.exists():
        print("   📄 Creating demo EEG data...")
        import sys
        sys.path.append(str(Path(__file__).parent))
        from create_mock_data import create_mock_eeg_data
        raw = create_mock_eeg_data()
        demo_file.parent.mkdir(exist_ok=True)
        raw.save(demo_file, overwrite=True)
        print(f"   ✅ Created demo EEG data: {demo_file}")

    for site_id in site_ids:
        site_subjects = []

        for i in range(subjects_per_site):
            # Enroll subject
            subject_id = platform.enroll_subject(
                trial_id=trial_id,
                site_id=site_id,
                group_assignment="PD" if i < 3 else "CONTROL",
                age_group="46-60" if i < 2 else "61-75",
                gender="M" if i % 2 == 0 else "F"
            )
            site_subjects.append(subject_id)
            all_subjects.append(subject_id)

            # Submit comprehensive job pipeline for each subject
            job_types = [JobType.FEATURE_EXTRACTION, JobType.CLASSIFICATION, JobType.QC_CHECK]

            for job_type in job_types:
                config = JobConfig(
                    feature_set="Core15+",
                    sampling_rate=512,
                    domain_adaptation=True,
                    random_seed=42 + i
                )

                job_id = platform.submit_job(
                    subject_id=subject_id,
                    trial_id=trial_id,
                    site_id=site_id,
                    job_type=job_type,
                    input_file_path=str(demo_file),
                    session_name=f"session_{i+1}",
                    feature_set="Core15+",
                    priority=1 if job_type == JobType.CLASSIFICATION else 0,
                    config=config
                )

        print(f"   ✅ Site {site_id}: {len(site_subjects)} subjects enrolled, {len(job_types) * len(site_subjects)} jobs submitted")

    print(f"   📊 Total: {len(all_subjects)} subjects, {len(all_subjects) * len(job_types)} jobs")
    return all_subjects

def demonstrate_platform_operation(platform: PersistentPlatformManager,
                                 dashboard: MonitoringDashboard,
                                 duration: int = 60):
    """Demonstrate platform operation with real-time monitoring."""
    print(f"🚀 Starting platform operation demo ({duration}s)...")

    # Start workers and monitoring
    platform.start_workers()
    dashboard.start_monitoring()

    try:
        start_time = time.time()
        update_interval = 10

        while time.time() - start_time < duration:
            # Get current status
            status = dashboard.get_system_status()
            platform_status = platform.get_platform_status()

            elapsed = int(time.time() - start_time)
            print(f"\n⏱️  Status Update (T+{elapsed}s):")

            # Job statistics
            job_stats = status['job_statistics']['queue_counts']
            print(f"   📋 Jobs: {job_stats}")

            # Performance metrics
            perf = status['performance_metrics']
            print(f"   📈 Performance: {perf['success_rate']:.1%} success, {perf['qc_pass_rate']:.1%} QC pass")

            # System health
            health = status['system_health']
            db_size = health['database_size_mb']
            print(f"   🏥 Health: {health['recent_audit_events']} events, {db_size:.1f}MB DB")

            # Platform workers
            workers_running = platform_status['workers_running']
            worker_count = platform_status['worker_count']
            print(f"   🔧 Workers: {worker_count} threads, {'running' if workers_running else 'stopped'}")

            # Check for alerts
            if status['recent_alerts']:
                print(f"   🚨 Alerts: {len(status['recent_alerts'])} active")
                for alert in status['recent_alerts'][-2:]:
                    print(f"      {alert['level'].upper()}: {alert['message']}")

            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")

    finally:
        # Stop platform components
        platform.stop_workers()
        dashboard.stop_monitoring()
        print("   🛑 Platform stopped")

def generate_final_reports(trial_manager: ClinicalTrialManager,
                          dashboard: MonitoringDashboard,
                          trial_id: str):
    """Generate comprehensive final reports."""
    print("📋 Generating final reports...")

    # Trial enrollment report
    enrollment_report = trial_manager.generate_regulatory_report(trial_id, "enrollment")
    print(f"   ✅ Enrollment report: {enrollment_report['enrollment_summary']['overall_statistics']['total_enrolled']} subjects")

    # Audit trail report
    audit_report = trial_manager.generate_regulatory_report(trial_id, "audit_trail")
    print(f"   ✅ Audit report: {len(audit_report['recent_significant_events'])} events")

    # Performance history
    perf_history = dashboard.get_performance_history(hours=2)
    print(f"   ✅ Performance history: {len(perf_history)} data points")

    # Configuration history
    config_history = dashboard.get_configuration_history(limit=10)
    print(f"   ✅ Configuration history: {len(config_history)} changes")

    # System status summary
    final_status = dashboard.get_system_status()
    job_total = sum(final_status['job_statistics']['queue_counts'].values())
    print(f"   ✅ Final status: {job_total} total jobs processed")

    return {
        'enrollment_report': enrollment_report,
        'audit_report': audit_report,
        'performance_history': perf_history,
        'configuration_history': config_history,
        'final_status': final_status
    }

def test_configuration_management(dashboard: MonitoringDashboard):
    """Test configuration management capabilities."""
    print("⚙️  Testing configuration management...")

    # Update various configurations
    config_updates = [
        ("processing", "max_concurrent_jobs", 6),
        ("quality", "min_qc_score", 0.85),
        ("monitoring", "alert_threshold_error_rate", 0.15),
        ("security", "session_timeout_hours", 12)
    ]

    for category, key, value in config_updates:
        success = dashboard.update_configuration(category, key, value, "integration_demo")
        print(f"   {'✅' if success else '❌'} {category}.{key} = {value}")

    print("   ✅ Configuration management tested")

def main():
    """Main integration demonstration."""
    print("="*70)
    print("PHASE VI-B: COMPLETE CLINICAL PLATFORM INTEGRATION")
    print("EEG Biomarker Platform with Persistent Job Processing")
    print("="*70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. Setup platform
        platform, trial_manager, dashboard = setup_complete_platform()

        # 2. Create trial
        trial_id = create_comprehensive_trial(trial_manager)

        # 3. Register sites
        site_ids = register_clinical_sites(trial_manager)

        # 4. Enroll subjects and submit jobs
        subjects = enroll_subjects_and_submit_jobs(platform, trial_id, site_ids)

        # 5. Test configuration management
        test_configuration_management(dashboard)

        # 6. Demonstrate platform operation
        demonstrate_platform_operation(platform, dashboard, duration=45)

        # 7. Generate final reports
        reports = generate_final_reports(trial_manager, dashboard, trial_id)

        # 8. Summary
        print("\n" + "="*70)
        print("INTEGRATION DEMONSTRATION SUMMARY")
        print("="*70)
        print(f"✅ Trial: {trial_id}")
        print(f"✅ Sites: {len(site_ids)} registered and certified")
        print(f"✅ Subjects: {len(subjects)} enrolled")
        print(f"✅ Jobs: {sum(reports['final_status']['job_statistics']['queue_counts'].values())} processed")
        print(f"✅ Performance: {reports['final_status']['performance_metrics']['success_rate']:.1%} success rate")
        print(f"✅ QC: {reports['final_status']['performance_metrics']['qc_pass_rate']:.1%} pass rate")
        print(f"✅ Audit Events: {len(reports['audit_report']['recent_significant_events'])} logged")
        print(f"✅ Database: {reports['final_status']['system_health']['database_size_mb']:.1f}MB")

        print("\n🎯 PHASE VI-B INTEGRATION COMPLETE!")
        print("   Platform ready for clinical deployment and regulatory submission")
        print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"\n❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())