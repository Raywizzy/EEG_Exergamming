#!/usr/bin/env python3
"""LOSO Results Visualization for Publication-Ready Figures.

Generates publication-quality plots and tables from LOSO validation results.
Phase IV Clinical-Trial Grade Implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import json

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def load_loso_results(results_file: str) -> pd.DataFrame:
    """Load LOSO results from JSON or CSV file.

    Args:
        results_file: Path to LOSO results file

    Returns:
        DataFrame with LOSO fold results
    """
    results_path = Path(results_file)

    if not results_path.exists():
        raise FileNotFoundError(f"LOSO results file not found: {results_file}")

    if results_path.suffix == '.json':
        with open(results_path, 'r') as f:
            data = json.load(f)

        # Extract fold results
        fold_results = data.get('fold_results', {})

        # Convert to DataFrame
        rows = []
        for test_site, metrics in fold_results.items():
            row = {
                'test_site': test_site,
                'train_sites': ', '.join(metrics.get('train_datasets', [])),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                'accuracy': metrics.get('accuracy', 0),
                'n_test_samples': metrics.get('n_test_samples', 0),
                'n_pd_test': metrics.get('n_pd_test', 0),
                'n_hc_test': metrics.get('n_hc_test', 0)
            }
            rows.append(row)

        df = pd.DataFrame(rows)

    elif results_path.suffix == '.csv':
        df = pd.read_csv(results_path)
    else:
        raise ValueError(f"Unsupported file format: {results_path.suffix}")

    return df

def plot_loso_bar_chart(df: pd.DataFrame, output_dir: Path) -> str:
    """Create publication-ready LOSO bar chart with confidence intervals.

    Args:
        df: DataFrame with LOSO results
        output_dir: Directory to save plots

    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Calculate statistics
    mean_ba = df['balanced_accuracy'].mean()
    std_ba = df['balanced_accuracy'].std()
    n_folds = len(df)
    sem_ba = std_ba / np.sqrt(n_folds)
    ci_95 = 1.96 * sem_ba

    # Individual fold results
    x_pos = range(len(df))
    bars = ax.bar(x_pos, df['balanced_accuracy'],
                  alpha=0.7, color='steelblue', edgecolor='navy', linewidth=1.5,
                  label='Individual Folds')

    # Add value labels on bars
    for i, (bar, ba) in enumerate(zip(bars, df['balanced_accuracy'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ba:.3f}', ha='center', va='bottom', fontweight='bold')

    # Mean line with CI
    ax.axhline(y=mean_ba, color='red', linestyle='-', linewidth=2,
               label=f'Mean: {mean_ba:.3f} ± {ci_95:.3f}')
    ax.axhline(y=mean_ba + ci_95, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=mean_ba - ci_95, color='red', linestyle='--', alpha=0.7)

    # Clinical threshold line
    clinical_threshold = 0.65
    ax.axhline(y=clinical_threshold, color='green', linestyle=':', linewidth=2,
               label=f'Clinical Threshold: {clinical_threshold:.2f}')

    # Chance performance line
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.8,
               label='Chance Performance: 0.50')

    # Formatting
    ax.set_xlabel('Test Site (Left-Out Fold)', fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontweight='bold')
    ax.set_title('Leave-One-Site-Out Cross-Validation Results\nPhase IV Multi-Site EEG Biomarker Validation',
                 fontweight='bold', pad=20)

    # X-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([site.replace('ds', 'Dataset ') for site in df['test_site']],
                       rotation=45, ha='right')

    # Y-axis limits
    y_min = max(0.4, min(df['balanced_accuracy']) - 0.05)
    y_max = min(1.0, max(df['balanced_accuracy']) + 0.1)
    ax.set_ylim(y_min, y_max)

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # Add statistics box
    stats_text = (f'n = {n_folds} folds\n'
                  f'Mean BA = {mean_ba:.3f}\n'
                  f'95% CI = [{mean_ba - ci_95:.3f}, {mean_ba + ci_95:.3f}]\n'
                  f'SD = {std_ba:.3f}')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f'loso_validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return str(output_file)

def create_results_table(df: pd.DataFrame, output_dir: Path) -> str:
    """Create publication-ready results table.

    Args:
        df: DataFrame with LOSO results
        output_dir: Directory to save table

    Returns:
        Path to saved table image
    """
    # Calculate statistics
    mean_ba = df['balanced_accuracy'].mean()
    std_ba = df['balanced_accuracy'].std()
    sem_ba = std_ba / np.sqrt(len(df))
    ci_95 = 1.96 * sem_ba

    # Prepare table data
    table_data = []

    # Individual fold results
    for _, row in df.iterrows():
        table_data.append([
            row['test_site'].replace('ds', 'Dataset '),
            ', '.join([site.replace('ds', 'Dataset ') for site in row['train_sites'].split(', ')]),
            f"{row['balanced_accuracy']:.3f}",
            f"{row['accuracy']:.3f}",
            f"{row['n_test_samples']:.0f}",
            f"{row['n_pd_test']:.0f}",
            f"{row['n_hc_test']:.0f}"
        ])

    # Summary row
    table_data.append([
        '**Overall**',
        '**All Sites**',
        f"**{mean_ba:.3f} ± {ci_95:.3f}**",
        f"**{df['accuracy'].mean():.3f}**",
        f"**{df['n_test_samples'].sum():.0f}**",
        f"**{df['n_pd_test'].sum():.0f}**",
        f"**{df['n_hc_test'].sum():.0f}**"
    ])

    # Column headers
    headers = ['Test Site', 'Training Sites', 'Balanced Accuracy', 'Accuracy',
               'Total N', 'PD N', 'HC N']

    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(table_data) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Header styling
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Summary row styling
    for i in range(len(headers)):
        table[(len(table_data), i)].set_facecolor('#E7E6E6')
        table[(len(table_data), i)].set_text_props(weight='bold')

    # Alternating row colors
    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')

    # Title
    plt.title('Leave-One-Site-Out Cross-Validation Results\nPhase IV Multi-Site EEG Biomarker Validation',
              fontsize=16, fontweight='bold', pad=20)

    # Statistical summary
    stats_text = (f'Statistical Summary:\n'
                  f'• Mean Balanced Accuracy: {mean_ba:.3f} (95% CI: {mean_ba - ci_95:.3f}–{mean_ba + ci_95:.3f})\n'
                  f'• Standard Deviation: {std_ba:.3f}\n'
                  f'• Number of Folds: {len(df)}\n'
                  f'• Clinical Threshold (≥0.65): {"✓ Met" if mean_ba >= 0.65 else "✗ Not Met"}\n'
                  f'• Statistical Significance vs Chance: {"✓ p < 0.05" if mean_ba > 0.5 + ci_95 else "✗ p ≥ 0.05"}')

    plt.figtext(0.1, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    # Save table
    output_file = output_dir / f'loso_results_table_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return str(output_file)

def generate_statistical_report(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate statistical analysis report.

    Args:
        df: DataFrame with LOSO results
        output_dir: Directory to save report

    Returns:
        Path to saved report
    """
    # Calculate statistics
    mean_ba = df['balanced_accuracy'].mean()
    std_ba = df['balanced_accuracy'].std()
    n_folds = len(df)
    sem_ba = std_ba / np.sqrt(n_folds)
    ci_95 = 1.96 * sem_ba

    # One-sample t-test vs chance
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(df['balanced_accuracy'], 0.5)

    # Effect size (Cohen's d)
    cohens_d = (mean_ba - 0.5) / std_ba

    # Clinical significance
    clinical_threshold = 0.65
    meets_clinical = mean_ba >= clinical_threshold

    # Create report
    report = {
        'loso_statistical_analysis': {
            'timestamp': datetime.now().isoformat(),
            'n_folds': int(n_folds),
            'datasets_tested': df['test_site'].tolist(),
            'balanced_accuracy_stats': {
                'mean': float(mean_ba),
                'std': float(std_ba),
                'sem': float(sem_ba),
                'ci_95_lower': float(mean_ba - ci_95),
                'ci_95_upper': float(mean_ba + ci_95),
                'min': float(df['balanced_accuracy'].min()),
                'max': float(df['balanced_accuracy'].max())
            },
            'statistical_tests': {
                'one_sample_ttest_vs_chance': {
                    'null_hypothesis': 'mean_ba = 0.50',
                    'alternative': 'mean_ba ≠ 0.50',
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05),
                    'interpretation': 'Significantly above chance' if p_value < 0.05 else 'Not significantly different from chance'
                }
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': 'Large effect' if abs(cohens_d) >= 0.8 else 'Medium effect' if abs(cohens_d) >= 0.5 else 'Small effect'
            },
            'clinical_significance': {
                'clinical_threshold': clinical_threshold,
                'meets_threshold': bool(meets_clinical),
                'margin': float(mean_ba - clinical_threshold),
                'interpretation': 'Clinically meaningful' if meets_clinical else 'Below clinical threshold'
            },
            'sample_characteristics': {
                'total_subjects_tested': int(df['n_test_samples'].sum()),
                'total_pd_subjects': int(df['n_pd_test'].sum()),
                'total_hc_subjects': int(df['n_hc_test'].sum()),
                'pd_hc_ratio': float(df['n_pd_test'].sum() / df['n_hc_test'].sum()) if df['n_hc_test'].sum() > 0 else None
            }
        }
    }

    # Save report
    output_file = output_dir / f'loso_statistical_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    return str(output_file)

def main():
    parser = argparse.ArgumentParser(description="Generate LOSO validation plots and tables")
    parser.add_argument('--results', required=True, help='Path to LOSO results file (JSON or CSV)')
    parser.add_argument('--output-dir', default='results/loso_plots', help='Output directory for plots')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png', help='Output format')

    args = parser.parse_args()

    print("="*80)
    print("LOSO VALIDATION RESULTS VISUALIZATION")
    print("Phase IV Clinical-Trial Grade Implementation")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load results
        print(f"📊 Loading LOSO results from: {args.results}")
        df = load_loso_results(args.results)
        print(f"✅ Loaded {len(df)} LOSO folds")

        # Generate visualizations
        print("\n🎨 Generating publication-ready visualizations...")

        # Bar chart
        bar_chart_file = plot_loso_bar_chart(df, output_dir)
        print(f"📊 Bar chart saved: {bar_chart_file}")

        # Results table
        table_file = create_results_table(df, output_dir)
        print(f"📋 Results table saved: {table_file}")

        # Statistical report
        stats_file = generate_statistical_report(df, output_dir)
        print(f"📈 Statistical report saved: {stats_file}")

        # Summary statistics
        mean_ba = df['balanced_accuracy'].mean()
        std_ba = df['balanced_accuracy'].std()
        ci_95 = 1.96 * std_ba / np.sqrt(len(df))

        print(f"\n🎯 SUMMARY STATISTICS:")
        print(f"   Mean Balanced Accuracy: {mean_ba:.3f} ± {ci_95:.3f}")
        print(f"   Range: {df['balanced_accuracy'].min():.3f} - {df['balanced_accuracy'].max():.3f}")
        print(f"   Clinical Threshold (≥0.65): {'✅ Met' if mean_ba >= 0.65 else '❌ Not Met'}")
        print(f"   Total Subjects: {df['n_test_samples'].sum()}")

        print(f"\n✅ All visualizations generated successfully!")
        print(f"📁 Output directory: {output_dir}")

    except Exception as e:
        print(f"❌ Error generating LOSO visualizations: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())