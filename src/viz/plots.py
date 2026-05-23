"""
External Validation Visualization Suite
Comprehensive plotting functions for external validation report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import scipy.stats as stats

def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: List[str] = None,
                         title: str = "Confusion Matrix - External Validation",
                         out_path: Path = None, dpi: int = 300,
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot confusion matrix with counts and percentages.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        labels: Class labels
        title: Plot title
        out_path: Output file path
        dpi: Figure DPI
        figsize: Figure size
    """
    setup_plot_style()

    if labels is None:
        labels = ['PD_REAL', 'PD_SHAM']

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=labels, yticklabels=labels)
    ax1.set_title('Counts')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=ax2,
                xticklabels=labels, yticklabels=labels)
    ax2.set_title('Proportions')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   title: str = "ROC Curve - External Validation",
                   out_path: Path = None, dpi: int = 300,
                   figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot ROC curve with AUC score.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        title: Plot title
        out_path: Output file path
        dpi: Figure DPI
        figsize: Figure size
    """
    setup_plot_style()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray,
                  title: str = "Precision-Recall Curve - External Validation",
                  out_path: Path = None, dpi: int = 300,
                  figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot Precision-Recall curve with Average Precision score.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        title: Plot title
        out_path: Output file path
        dpi: Figure DPI
        figsize: Figure size
    """
    setup_plot_style()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    from sklearn.metrics import average_precision_score
    ap_score = average_precision_score(y_true, y_prob)

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, 'b-', linewidth=2,
             label=f'PR Curve (AP = {ap_score:.3f})')
    plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.7,
                label=f'Random Classifier (AP = {baseline:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_calibration_curve(fraction_pos: np.ndarray, mean_pred: np.ndarray,
                          title: str = "Calibration Curve - External Validation",
                          out_path: Path = None, dpi: int = 300,
                          figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot calibration curve.

    Args:
        fraction_pos: Fraction of positives in each bin
        mean_pred: Mean predicted probability in each bin
        title: Plot title
        out_path: Output file path
        dpi: Figure DPI
        figsize: Figure size
    """
    setup_plot_style()

    plt.figure(figsize=figsize)
    plt.plot(mean_pred, fraction_pos, 'bo-', linewidth=2, markersize=8,
             label='Model Calibration')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_metric_table(metrics: Dict[str, float],
                     title: str = "External Validation Metrics",
                     out_path: Path = None, dpi: int = 300,
                     figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot metrics as a formatted table.

    Args:
        metrics: Dictionary of metric names and values
        title: Plot title
        out_path: Output file path
        dpi: Figure DPI
        figsize: Figure size
    """
    setup_plot_style()

    # Select key metrics for display
    key_metrics = [
        'balanced_accuracy', 'accuracy', 'sensitivity', 'specificity',
        'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'brier_score'
    ]

    # Create display data
    display_names = {
        'balanced_accuracy': 'Balanced Accuracy',
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity (Recall)',
        'specificity': 'Specificity',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'roc_auc': 'ROC AUC',
        'pr_auc': 'PR AUC',
        'brier_score': 'Brier Score'
    }

    table_data = []
    for metric in key_metrics:
        if metric in metrics:
            name = display_names.get(metric, metric)
            value = metrics[metric]
            table_data.append([name, f"{value:.4f}"])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Value'],
                    cellLoc='left',
                    loc='center')

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Color header
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows alternately
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f5f5f5')

    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_shift(X_external: pd.DataFrame,
                      training_stats: Dict[str, Dict[str, float]],
                      top_k: int = 10,
                      title: str = "Feature Distribution Shift Analysis",
                      out_path: Path = None, dpi: int = 300,
                      figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot feature distribution shift between training and external data.

    Args:
        X_external: External dataset features
        training_stats: Training feature statistics
        top_k: Number of top features to plot
        title: Plot title
        out_path: Output file path
        dpi: Figure DPI
        figsize: Figure size
    """
    setup_plot_style()

    # Get top k features
    feature_names = list(training_stats.keys())[:top_k]
    available_features = [f for f in feature_names if f in X_external.columns]

    if not available_features:
        print("No matching features found for shift analysis")
        return

    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(available_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Get data
        external_data = X_external[feature].values
        train_stats = training_stats[feature]

        # Plot histograms
        ax.hist(external_data, bins=20, alpha=0.7, label='External',
                density=True, color='orange')

        # Add training distribution (approximate as normal)
        x_range = np.linspace(train_stats['min'], train_stats['max'], 100)
        train_density = stats.norm.pdf(x_range, train_stats['mean'], train_stats['std'])
        ax.plot(x_range, train_density, 'b-', linewidth=2, label='Training')

        # Add statistics
        ext_mean = np.mean(external_data)
        shift = (ext_mean - train_stats['mean']) / train_stats['std']

        ax.axvline(train_stats['mean'], color='blue', linestyle='--', alpha=0.7)
        ax.axvline(ext_mean, color='orange', linestyle='--', alpha=0.7)

        ax.set_title(f"{feature}\nShift: {shift:.2f}σ", fontsize=9)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_performance_comparison(phase6_metrics: Dict[str, float],
                               external_metrics: Dict[str, float],
                               title: str = "Internal vs External Performance",
                               out_path: Path = None, dpi: int = 300,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot comparison between Phase 6 and external validation performance.

    Args:
        phase6_metrics: Phase 6 performance metrics
        external_metrics: External validation metrics
        title: Plot title
        out_path: Output file path
        dpi: Figure DPI
        figsize: Figure size
    """
    setup_plot_style()

    # Key metrics to compare
    key_metrics = ['balanced_accuracy', 'roc_auc', 'sensitivity', 'specificity',
                   'precision', 'f1_score']

    metric_names = []
    phase6_values = []
    external_values = []

    for metric in key_metrics:
        if metric in phase6_metrics and metric in external_metrics:
            metric_names.append(metric.replace('_', ' ').title())
            phase6_values.append(phase6_metrics[metric])
            external_values.append(external_metrics[metric])

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, phase6_values, width, label='Phase 6 (Internal CV)',
                   color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, external_values, width, label='External Validation',
                   color='lightcoral', alpha=0.8)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()