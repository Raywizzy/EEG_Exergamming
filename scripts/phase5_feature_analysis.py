#!/usr/bin/env python3
"""
Phase 5 Feature Analysis: Quick inspection of beta burst feature separation
"""

import pandas as pd
import numpy as np
from scipy import stats

def main():
    # Load beta burst features
    df = pd.read_csv("results/phase5_beta_bursts/beta_burst_features.csv")

    print("=" * 60)
    print("BETA BURST FEATURE ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"PD_REAL: {len(df[df['condition'] == 'PD_REAL'])}")
    print(f"PD_SHAM: {len(df[df['condition'] == 'PD_SHAM'])}")
    print()

    # Key features to analyze
    key_features = [
        'rate_per_min',
        'motor_posterior_rate_ratio',
        'duty_cycle',
        'mean_duration_ms'
    ]

    print("KEY FEATURE SEPARABILITY:")
    print("-" * 60)

    for feature in key_features:
        pd_real_data = df[df['condition'] == 'PD_REAL'][feature]
        pd_sham_data = df[df['condition'] == 'PD_SHAM'][feature]

        # Calculate statistics
        real_mean = pd_real_data.mean()
        sham_mean = pd_sham_data.mean()

        # Cohen's d
        pooled_std = np.sqrt(((len(pd_real_data) - 1) * pd_real_data.var() +
                             (len(pd_sham_data) - 1) * pd_sham_data.var()) /
                            (len(pd_real_data) + len(pd_sham_data) - 2))
        cohens_d = (real_mean - sham_mean) / pooled_std

        # T-test
        t_stat, p_value = stats.ttest_ind(pd_real_data, pd_sham_data)

        print(f"{feature}:")
        print(f"  PD_REAL: {real_mean:.3f} ± {pd_real_data.std():.3f}")
        print(f"  PD_SHAM: {sham_mean:.3f} ± {pd_sham_data.std():.3f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  p-value: {p_value:.6f}")

        # Effect size interpretation
        if abs(cohens_d) >= 0.8:
            effect = "LARGE"
        elif abs(cohens_d) >= 0.5:
            effect = "MEDIUM"
        elif abs(cohens_d) >= 0.2:
            effect = "SMALL"
        else:
            effect = "NEGLIGIBLE"

        print(f"  Effect: {effect}")
        print()

    print("=" * 60)
    print("PRELIMINARY ASSESSMENT:")

    # Check motor_posterior_rate_ratio specifically
    ratio_feature = 'motor_posterior_rate_ratio'
    pd_real_ratio = df[df['condition'] == 'PD_REAL'][ratio_feature]
    pd_sham_ratio = df[df['condition'] == 'PD_SHAM'][ratio_feature]
    ratio_cohens_d = (pd_real_ratio.mean() - pd_sham_ratio.mean()) / np.sqrt(
        ((len(pd_real_ratio) - 1) * pd_real_ratio.var() +
         (len(pd_sham_ratio) - 1) * pd_sham_ratio.var()) /
        (len(pd_real_ratio) + len(pd_sham_ratio) - 2)
    )

    if ratio_cohens_d > 1.0:
        print("🎯 STRONG DISCRIMINATIVE SIGNAL DETECTED!")
        print(f"   Motor/Posterior ratio shows Cohen's d = {ratio_cohens_d:.3f}")
        print("   This suggests beta burst features may achieve >70% accuracy")
        print("   Ready for Phase 6 model training!")
    else:
        print("⚠️  Moderate signal detected")
        print("   Features show separation but may need optimization")

    print("=" * 60)

if __name__ == "__main__":
    main()