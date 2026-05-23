#!/usr/bin/env python3
"""
Phase 6 Integrity Check Script
==============================

This script performs comprehensive integrity checks on Phase 6 results to identify
potential issues with statistical validation, cross-validation procedures, and
data quality.

Author: EEG Exergaming Development Team
Date: 2025-09-14
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/phase6_integrity_check_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_phase6_results():
    """Load all Phase 6 result files"""
    results_dir = Path("results/phase6_burst_models")

    if not results_dir.exists():
        logger.error(f"Phase 6 results directory not found: {results_dir}")
        return None

    files = {
        'cv_results': results_dir / 'cv_results.json',
        'statistical_validation': results_dir / 'statistical_validation.json'
    }

    data = {}
    for key, file_path in files.items():
        if file_path.exists():
            with open(file_path, 'r') as f:
                data[key] = json.load(f)
            logger.info(f"Loaded {key}: {file_path}")
        else:
            logger.error(f"Missing required file: {file_path}")
            return None

    return data

def check_cross_validation_integrity(cv_results):
    """Check cross-validation integrity"""
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION INTEGRITY CHECK")
    logger.info("=" * 60)

    issues = []

    # Check subject-wise split integrity
    total_subjects = 31
    for model_name, model_results in cv_results['cv_results'].items():
        logger.info(f"\nChecking {model_name} cross-validation:")

        fold_results = model_results['fold_results']

        # Check total subjects across folds
        total_test_subjects = sum(fold['test_subjects'] for fold in fold_results)
        if total_test_subjects != total_subjects:
            issue = f"{model_name}: Total test subjects ({total_test_subjects}) != expected ({total_subjects})"
            issues.append(issue)
            logger.error(f"  ❌ {issue}")
        else:
            logger.info(f"  ✅ Subject count consistent: {total_test_subjects} subjects")

        # Check fold balance
        test_subject_counts = [fold['test_subjects'] for fold in fold_results]
        logger.info(f"  📊 Test subjects per fold: {test_subject_counts}")

        # Check performance consistency
        bas = [fold['balanced_accuracy'] for fold in fold_results]
        ba_mean = np.mean(bas)
        ba_std = np.std(bas)

        logger.info(f"  📈 BA per fold: {[f'{ba:.3f}' for ba in bas]}")
        logger.info(f"  📊 BA mean: {ba_mean:.3f} ± {ba_std:.3f}")

        # Check for suspiciously high performance
        if ba_mean > 0.95:
            issue = f"{model_name}: Suspiciously high BA ({ba_mean:.3f}) - possible overfitting"
            issues.append(issue)
            logger.warning(f"  ⚠️  {issue}")

    return issues

def check_statistical_validation_integrity(stat_results):
    """Check statistical validation integrity"""
    logger.info("\n" + "=" * 60)
    logger.info("STATISTICAL VALIDATION INTEGRITY CHECK")
    logger.info("=" * 60)

    issues = []

    # Check permutation test results
    perm_scores = stat_results['permutation_scores']
    perm_mean = stat_results['permutation_mean']
    perm_std = stat_results['permutation_std']
    p_value = stat_results['p_value']
    observed_score = stat_results['observed_score']

    logger.info(f"Observed score: {observed_score:.6f}")
    logger.info(f"Permutation mean: {perm_mean:.6f}")
    logger.info(f"Permutation std: {perm_std:.2e}")
    logger.info(f"P-value: {p_value:.6f}")

    # Check for identical permutation scores (major bug indicator)
    unique_scores = len(set(perm_scores))
    total_scores = len(perm_scores)

    logger.info(f"Unique permutation scores: {unique_scores}/{total_scores}")

    if unique_scores == 1:
        issue = "CRITICAL: All permutation scores identical - labels not shuffled!"
        issues.append(issue)
        logger.error(f"  ❌ {issue}")
        logger.error(f"  📊 All {total_scores} scores = {perm_scores[0]:.6f}")
    elif unique_scores < total_scores * 0.8:
        issue = f"WARNING: Low permutation score diversity ({unique_scores}/{total_scores})"
        issues.append(issue)
        logger.warning(f"  ⚠️  {issue}")
    else:
        logger.info(f"  ✅ Permutation score diversity: {unique_scores}/{total_scores}")

    # Check permutation standard deviation
    if perm_std < 1e-10:
        issue = f"CRITICAL: Permutation std too small ({perm_std:.2e}) - no randomization"
        issues.append(issue)
        logger.error(f"  ❌ {issue}")
    elif perm_std < 0.01:
        issue = f"WARNING: Very low permutation std ({perm_std:.6f})"
        issues.append(issue)
        logger.warning(f"  ⚠️  {issue}")
    else:
        logger.info(f"  ✅ Permutation std reasonable: {perm_std:.6f}")

    # Check p-value
    if p_value >= 1.0:
        issue = "CRITICAL: p-value = 1.0 indicates no permutation effect"
        issues.append(issue)
        logger.error(f"  ❌ {issue}")
    elif p_value > 0.05:
        issue = f"WARNING: p-value ({p_value:.3f}) > 0.05 - not significant"
        issues.append(issue)
        logger.warning(f"  ⚠️  {issue}")
    else:
        logger.info(f"  ✅ P-value significant: {p_value:.6f}")

    # Check bootstrap results
    bootstrap_scores = stat_results['bootstrap_scores']
    bootstrap_mean = stat_results['bootstrap_mean']
    bootstrap_std = stat_results['bootstrap_std']
    ci_95 = stat_results['confidence_interval_95']

    logger.info(f"\nBootstrap results:")
    logger.info(f"  Bootstrap mean: {bootstrap_mean:.6f}")
    logger.info(f"  Bootstrap std: {bootstrap_std:.6f}")
    logger.info(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")

    # Check bootstrap diversity
    unique_bootstrap = len(set(np.round(bootstrap_scores, 6)))
    total_bootstrap = len(bootstrap_scores)

    if unique_bootstrap < total_bootstrap * 0.8:
        issue = f"WARNING: Low bootstrap diversity ({unique_bootstrap}/{total_bootstrap})"
        issues.append(issue)
        logger.warning(f"  ⚠️  {issue}")
    else:
        logger.info(f"  ✅ Bootstrap diversity: {unique_bootstrap}/{total_bootstrap}")

    return issues

def check_performance_plausibility(cv_results):
    """Check if performance results are plausible"""
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE PLAUSIBILITY CHECK")
    logger.info("=" * 60)

    issues = []

    performances = cv_results['model_performances']

    for model_name, performance in performances.items():
        logger.info(f"\n{model_name}: {performance:.3f}")

        # Check for unrealistic high performance
        if performance > 0.95:
            issue = f"{model_name}: Unrealistically high performance ({performance:.3f}) - possible data leakage"
            issues.append(issue)
            logger.error(f"  ❌ {issue}")
        elif performance > 0.90:
            issue = f"{model_name}: Very high performance ({performance:.3f}) - verify no overfitting"
            issues.append(issue)
            logger.warning(f"  ⚠️  {issue}")
        else:
            logger.info(f"  ✅ Performance reasonable: {performance:.3f}")

        # Check for unusually low performance
        if performance < 0.55:
            issue = f"{model_name}: Low performance ({performance:.3f}) - barely above chance"
            issues.append(issue)
            logger.warning(f"  ⚠️  {issue}")

    # Check performance ranking
    sorted_perf = sorted(performances.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"\nPerformance ranking:")
    for i, (model, perf) in enumerate(sorted_perf, 1):
        logger.info(f"  {i}. {model}: {perf:.3f}")

    return issues

def generate_integrity_report(issues):
    """Generate final integrity report"""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6 INTEGRITY ASSESSMENT SUMMARY")
    logger.info("=" * 60)

    if not issues:
        logger.info("✅ ALL INTEGRITY CHECKS PASSED")
        status = "PASS"
    else:
        logger.error(f"❌ FOUND {len(issues)} INTEGRITY ISSUES:")
        for i, issue in enumerate(issues, 1):
            logger.error(f"  {i}. {issue}")
        status = "FAIL"

    # Determine severity
    critical_issues = [i for i in issues if "CRITICAL" in i]
    warning_issues = [i for i in issues if "WARNING" in i]

    logger.info(f"\nIssue breakdown:")
    logger.info(f"  Critical issues: {len(critical_issues)}")
    logger.info(f"  Warning issues: {len(warning_issues)}")

    if critical_issues:
        logger.error("\n🚨 CRITICAL ISSUES REQUIRE IMMEDIATE ATTENTION:")
        for issue in critical_issues:
            logger.error(f"  • {issue}")

    return {
        'status': status,
        'total_issues': len(issues),
        'critical_issues': len(critical_issues),
        'warning_issues': len(warning_issues),
        'issues': issues
    }

def main():
    """Main integrity check execution"""
    logger.info("================================================================================")
    logger.info("PHASE 6 INTEGRITY CHECK STARTING")
    logger.info("================================================================================")
    logger.info(f"Start time: {pd.Timestamp.now().isoformat()}")

    # Load Phase 6 results
    data = load_phase6_results()
    if data is None:
        logger.error("Failed to load Phase 6 results - aborting integrity check")
        return

    all_issues = []

    # Run integrity checks
    all_issues.extend(check_cross_validation_integrity(data['cv_results']))
    all_issues.extend(check_statistical_validation_integrity(data['statistical_validation']))
    all_issues.extend(check_performance_plausibility(data['cv_results']))

    # Generate final report
    report = generate_integrity_report(all_issues)

    # Save report
    report_path = Path("results") / "phase6_integrity_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': pd.Timestamp.now().isoformat(),
            'phase': 'Phase 6',
            'assessment': 'Integrity Check',
            **report
        }, f, indent=2)

    logger.info(f"\nIntegrity report saved to: {report_path}")
    logger.info("================================================================================")
    logger.info("PHASE 6 INTEGRITY CHECK COMPLETED")
    logger.info("================================================================================")

    return report['status'] == "PASS"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)