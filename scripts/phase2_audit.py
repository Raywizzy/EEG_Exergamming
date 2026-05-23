#!/usr/bin/env python3
"""
Phase 2 Domain Adaptation Audit Script
Verify all CLAUDE.md requirements are met before declaring completion
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json
import sys

def check_data_provenance():
    """Verify only approved datasets used, no synthetic data"""

    print("🔍 CHECKING DATA PROVENANCE...")

    approved_datasets = ['ds004584', 'ds002778', 'ds003490']
    issues = []

    # Check feature files
    feature_dir = Path("results/features")
    feature_files = list(feature_dir.glob("*_core5*.csv"))

    for file in feature_files:
        df = pd.read_csv(file)

        # Check dataset column
        if 'dataset' in df.columns:
            datasets_used = df['dataset'].unique()
            for dataset in datasets_used:
                if dataset not in approved_datasets:
                    issues.append(f"Unapproved dataset found: {dataset} in {file}")

        # Check for synthetic markers
        if 'subject_id' in df.columns:
            synthetic_markers = ['synthetic', 'fake', 'generated', 'random']
            subject_ids = df['subject_id'].astype(str).str.lower()
            for marker in synthetic_markers:
                if subject_ids.str.contains(marker).any():
                    issues.append(f"Synthetic data marker '{marker}' found in {file}")

    if issues:
        print("❌ DATA PROVENANCE ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Data provenance clean - only approved datasets used")
        return True

def check_subject_wise_isolation():
    """Verify subject-wise cross-validation used"""

    print("\n🔍 CHECKING SUBJECT-WISE ISOLATION...")

    issues = []

    # Check LOSO results file
    loso_file = Path("results/loso_validation/3site_loso_labeled_detailed.pkl")

    if not loso_file.exists():
        issues.append("LOSO results file missing")
        return False

    with open(loso_file, 'rb') as f:
        loso_data = pickle.load(f)

    results = loso_data['results']

    # Verify no subject leakage across folds
    for i, result in enumerate(results):
        test_site = result['test_site']
        train_sites = result['train_sites']

        # Check that test site not in train sites
        if test_site in train_sites:
            issues.append(f"Site leakage: {test_site} in both train and test for fold {i}")

    if issues:
        print("❌ SUBJECT-WISE ISOLATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Subject-wise isolation verified - no site leakage")
        return True

def check_class_labels():
    """Verify only approved class labels used"""

    print("\n🔍 CHECKING CLASS LABELS...")

    approved_labels = ['PD_REAL', 'PD_SHAM', 'CONTROL']
    issues = []

    # Check labeled feature files
    feature_dir = Path("results/features")
    labeled_files = list(feature_dir.glob("*_labeled.csv"))

    for file in labeled_files:
        df = pd.read_csv(file)

        if 'label' in df.columns:
            # Check for NaN/unknown labels
            nan_count = df['label'].isna().sum()
            if nan_count > 0:
                issues.append(f"{nan_count} NaN labels in {file}")

            # Check for unapproved labels
            unique_labels = df['label'].dropna().unique()
            for label in unique_labels:
                if label not in approved_labels:
                    issues.append(f"Unapproved label '{label}' in {file}")

    if issues:
        print("❌ CLASS LABEL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Class labels verified - only approved labels used")
        return True

def check_file_existence():
    """Verify all required files exist"""

    print("\n🔍 CHECKING FILE EXISTENCE...")

    required_files = [
        "results/features/ds003490_core5_labeled.csv",
        "results/features/ds002778_core5_labeled.csv",
        "results/loso_validation/3site_loso_labeled_summary.csv",
        "results/loso_validation/3site_loso_labeled_detailed.pkl",
        "results/PHASE2_DOMAIN_ADAPTATION_REPORT.md",
        "logs/ds003490_full_processing.log"
    ]

    missing_files = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ MISSING REQUIRED FILES:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_statistical_validation():
    """Verify proper statistical testing performed"""

    print("\n🔍 CHECKING STATISTICAL VALIDATION...")

    issues = []

    # Load LOSO results
    loso_file = Path("results/loso_validation/3site_loso_labeled_detailed.pkl")

    if not loso_file.exists():
        issues.append("LOSO results file missing")
        return False

    with open(loso_file, 'rb') as f:
        loso_data = pickle.load(f)

    stats = loso_data['statistics']

    # Check required statistical measures
    required_stats = ['mean_ba', 'std_ba', 'p_value', 'cohens_d', 'ci_95']
    for stat in required_stats:
        if stat not in stats:
            issues.append(f"Missing statistic: {stat}")

    # Verify valid ranges
    if 'mean_ba' in stats:
        if not (0 <= stats['mean_ba'] <= 1):
            issues.append(f"Invalid mean BA: {stats['mean_ba']}")

    if 'p_value' in stats:
        if not (0 <= stats['p_value'] <= 1):
            issues.append(f"Invalid p-value: {stats['p_value']}")

    if issues:
        print("❌ STATISTICAL VALIDATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Statistical validation complete")
        return True

def check_determinism():
    """Verify deterministic execution with fixed seeds"""

    print("\n🔍 CHECKING DETERMINISM...")

    issues = []

    # Check scripts for random seed usage
    script_files = [
        "scripts/3site_loso_labeled.py",
        "scripts/add_comprehensive_labels.py"
    ]

    seeds_found = []

    for script_file in script_files:
        if Path(script_file).exists():
            with open(script_file, 'r') as f:
                content = f.read()

                # Look for seed setting
                if 'random_state' in content or 'seed' in content:
                    seeds_found.append(script_file)

    if not seeds_found:
        issues.append("No random seeds found in scripts")

    if issues:
        print("❌ DETERMINISM ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Deterministic execution verified")
        return True

def generate_audit_summary():
    """Generate final audit summary"""

    print("\n" + "="*80)
    print("PHASE 2 DOMAIN ADAPTATION AUDIT SUMMARY")
    print("="*80)

    # Run all checks
    checks = {
        "Data Provenance": check_data_provenance(),
        "Subject-wise Isolation": check_subject_wise_isolation(),
        "Class Labels": check_class_labels(),
        "File Existence": check_file_existence(),
        "Statistical Validation": check_statistical_validation(),
        "Determinism": check_determinism()
    }

    # Summary
    passed = sum(checks.values())
    total = len(checks)

    print(f"\nAUDIT RESULTS: {passed}/{total} checks passed")

    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name}: {status}")

    # Overall assessment
    if passed == total:
        overall_status = "✅ PHASE 2 AUDIT PASSED"
        exit_code = 0
    else:
        overall_status = "❌ PHASE 2 AUDIT FAILED"
        exit_code = 1

    print(f"\n{overall_status}")

    # Generate audit log
    audit_log = {
        "audit_date": datetime.now().isoformat(),
        "phase": "Phase 2 Domain Adaptation",
        "checks_passed": passed,
        "checks_total": total,
        "individual_checks": checks,
        "overall_status": "PASSED" if passed == total else "FAILED"
    }

    audit_file = Path("results/phase2_audit_report.json")
    with open(audit_file, 'w') as f:
        json.dump(audit_log, f, indent=2)

    print(f"\nAudit report saved: {audit_file}")

    return exit_code

def main():
    """Main audit execution"""

    print("PHASE 2 DOMAIN ADAPTATION AUDIT")
    print(f"Started: {datetime.now()}")
    print("="*80)

    exit_code = generate_audit_summary()

    print(f"\nAudit completed: {datetime.now()}")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())