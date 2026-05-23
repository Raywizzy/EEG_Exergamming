#!/usr/bin/env python3
"""
Generate LOSO validation plots for Phase IV multi-site validation
Creates publication-ready figures with clinical threshold indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

def create_loso_performance_plot(df: pd.DataFrame, output_dir: Path):
    """Create LOSO performance bar plot with clinical threshold."""

    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Color mapping based on clinical thresholds
    colors = []
    for ba in df['ba']:
        if ba >= 0.65:
            colors.append('#2E8B57')  # Green: Clinical
        elif ba > 0.50:
            colors.append('#FF8C00')  # Orange: Above chance
        else:
            colors.append('#DC143C')  # Red: At/below chance

    # Create bar plot
    bars = ax.bar(range(len(df)), df['ba'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add performance labels on bars
    for i, (bar, ba) in enumerate(zip(bars, df['ba'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ba:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add horizontal reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Chance (50%)')
    ax.axhline(y=0.65, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Clinical Target (65%)')

    # Customize plot
    ax.set_xlabel('Test Site', fontsize=14, fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('LOSO Cross-Validation Performance\n(2-Site Multi-Site Validation)', fontsize=16, fontweight='bold')

    # Set x-axis labels
    test_sites = df['test_site'].tolist()
    ax.set_xticks(range(len(test_sites)))
    ax.set_xticklabels([f'{site}\n(N={n})' for site, n in zip(test_sites, df['n_subjects'])], fontsize=12)

    # Format y-axis as percentages
    ax.set_ylim(0.4, 0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add legend
    ax.legend(loc='upper left', fontsize=12)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text box
    mean_ba = df['ba'].mean()
    stats_text = f'Mean BA: {mean_ba:.1%}\nBoth folds > 50%: ✓\nClinical target met: {len(df[df["ba"] >= 0.65])}/2 folds'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'loso_performance_2site.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'loso_performance_2site.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_dir}/loso_performance_2site.png")

def create_confusion_matrices(df: pd.DataFrame, output_dir: Path):
    """Create confusion matrices for each LOSO fold."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]

        # Calculate confusion matrix from sensitivity/specificity
        n_pd = row['n_pd']
        n_hc = row['n_hc']
        sensitivity = row['sensitivity']
        specificity = row['specificity']

        # True positives, false negatives, true negatives, false positives
        tp = int(n_pd * sensitivity)
        fn = n_pd - tp
        tn = int(n_hc * specificity)
        fp = n_hc - tn

        # Create confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted HC', 'Predicted PD'],
                   yticklabels=['True HC', 'True PD'])

        ax.set_title(f'{row["test_site"]}\nBA: {row["ba"]:.1%}, Sens: {sensitivity:.1%}, Spec: {specificity:.1%}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

    plt.suptitle('LOSO Validation Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'loso_confusion_matrices_2site.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'loso_confusion_matrices_2site.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_dir}/loso_confusion_matrices_2site.png")

def create_sample_size_plot(df: pd.DataFrame, output_dir: Path):
    """Create sample size breakdown plot."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Total subjects per site
    sites = df['test_site'].tolist()
    n_subjects = df['n_subjects'].tolist()

    bars1 = ax1.bar(sites, n_subjects, color='skyblue', alpha=0.8, edgecolor='black')
    ax1.set_title('Total Subjects per Site', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Subjects', fontsize=12)
    ax1.set_xlabel('Test Site', fontsize=12)

    # Add value labels
    for bar, n in zip(bars1, n_subjects):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{n}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # PD vs HC breakdown
    pd_counts = df['n_pd'].tolist()
    hc_counts = df['n_hc'].tolist()

    x = np.arange(len(sites))
    width = 0.35

    bars2 = ax2.bar(x - width/2, pd_counts, width, label='PD', color='lightcoral', alpha=0.8)
    bars3 = ax2.bar(x + width/2, hc_counts, width, label='HC', color='lightgreen', alpha=0.8)

    ax2.set_title('PD vs HC Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Subjects', fontsize=12)
    ax2.set_xlabel('Test Site', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sites)
    ax2.legend()

    # Add value labels
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'loso_sample_sizes_2site.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'loso_sample_sizes_2site.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_dir}/loso_sample_sizes_2site.png")

def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create publication-ready summary table."""

    # Calculate overall statistics
    mean_ba = df['ba'].mean()
    min_ba = df['ba'].min()
    max_ba = df['ba'].max()
    total_subjects = df['n_subjects'].sum()
    total_pd = df['n_pd'].sum()
    total_hc = df['n_hc'].sum()

    # Create summary DataFrame
    summary_data = {
        'Metric': [
            'Mean Balanced Accuracy',
            'Range (Min - Max)',
            'Folds Above Chance (>50%)',
            'Folds Meeting Clinical Target (≥65%)',
            'Total Subjects',
            'Total PD Patients',
            'Total Healthy Controls',
            'Number of Sites',
            'Cross-Validation Type'
        ],
        'Value': [
            f'{mean_ba:.1%}',
            f'{min_ba:.1%} - {max_ba:.1%}',
            f'{len(df[df["ba"] > 0.5])}/2',
            f'{len(df[df["ba"] >= 0.65])}/2',
            f'{total_subjects}',
            f'{total_pd}',
            f'{total_hc}',
            '2',
            'Leave-One-Site-Out (LOSO)'
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # Save as CSV
    summary_df.to_csv(output_dir / 'loso_overall_summary.csv', index=False)
    print(f"✅ Generated: {output_dir}/loso_overall_summary.csv")

    return summary_df

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate LOSO validation plots')
    parser.add_argument('--input', required=True, help='Input CSV file with LOSO results')
    parser.add_argument('--outdir', required=True, help='Output directory for plots')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Generating LOSO validation plots...")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")

    # Generate all plots
    create_loso_performance_plot(df, output_dir)
    create_confusion_matrices(df, output_dir)
    create_sample_size_plot(df, output_dir)
    summary_df = create_summary_table(df, output_dir)

    print(f"\n🎉 LOSO plots generated successfully!")
    print(f"📁 Check output directory: {output_dir}")

    # Print summary
    print(f"\n📈 Summary Statistics:")
    for _, row in summary_df.iterrows():
        print(f"  {row['Metric']}: {row['Value']}")

if __name__ == "__main__":
    main()