"""
Explainability Visualization System
Unified visualization interface for all explanation types
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import logging
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
import cv2

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ExplanationVisualizer:
    """Unified visualization system for all explanation types"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)

    def visualize_grad_cam(self, explanation: Dict[str, Any],
                          save_path: Optional[Path] = None) -> Tuple[plt.Figure, bytes]:
        """Visualize Grad-CAM explanation"""
        attribution = explanation['attribution']
        cam = attribution['cam']
        overlay = attribution['overlay']
        stats = attribution['statistics']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original CAM
        im1 = axes[0, 0].imshow(cam, cmap='jet', aspect='auto')
        axes[0, 0].set_title('Grad-CAM Heatmap')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')
        plt.colorbar(im1, ax=axes[0, 0])

        # Overlay
        axes[0, 1].imshow(overlay, aspect='auto')
        axes[0, 1].set_title('Spectrogram + Grad-CAM Overlay')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Frequency')

        # Frequency attribution profile
        freq_attribution = np.mean(cam, axis=1)
        axes[0, 2].plot(freq_attribution, range(len(freq_attribution)))
        axes[0, 2].set_title('Frequency Attribution Profile')
        axes[0, 2].set_xlabel('Attribution')
        axes[0, 2].set_ylabel('Frequency Bin')
        axes[0, 2].grid(True)

        # Temporal attribution profile
        time_attribution = np.mean(cam, axis=0)
        axes[1, 0].plot(time_attribution)
        axes[1, 0].set_title('Temporal Attribution Profile')
        axes[1, 0].set_xlabel('Time Bin')
        axes[1, 0].set_ylabel('Attribution')
        axes[1, 0].grid(True)

        # Attribution distribution
        axes[1, 1].hist(cam.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(stats['mean_attribution'], color='red', linestyle='--',
                          label=f"Mean: {stats['mean_attribution']:.3f}")
        axes[1, 1].axvline(stats['peak_attribution'], color='orange', linestyle='--',
                          label=f"Peak: {stats['peak_attribution']:.3f}")
        axes[1, 1].set_title('Attribution Distribution')
        axes[1, 1].set_xlabel('Attribution Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Statistics summary
        stats_text = (
            f"Peak Attribution: {stats['peak_attribution']:.3f}\n"
            f"Mean Attribution: {stats['mean_attribution']:.3f}\n"
            f"Attribution Std: {stats['attribution_std']:.3f}\n"
            f"Frequency Focus: {stats['frequency_focus']:.3f}\n"
            f"Temporal Focus: {stats['temporal_focus']:.3f}\n"
            f"Peak Freq Bin: {stats['peak_frequency_bin']}\n"
            f"Peak Time Bin: {stats['peak_time_bin']}"
        )
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 2].set_title('Attribution Statistics')
        axes[1, 2].axis('off')

        plt.suptitle(f"Grad-CAM Analysis - Subject: {explanation['subject_id']}\n"
                    f"Prediction: Class {explanation['prediction']['class']} "
                    f"(Confidence: {explanation['prediction']['confidence']:.3f})",
                    fontsize=14, y=0.98)

        plt.tight_layout()

        # Save as bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        buffer.close()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, image_bytes

    def visualize_attention(self, explanation: Dict[str, Any],
                           save_path: Optional[Path] = None) -> Tuple[plt.Figure, bytes]:
        """Visualize transformer attention explanation"""
        attention_data = explanation['attention']
        stats = attention_data['statistics']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # CLS attention pattern
        if attention_data['cls_attention'] is not None:
            cls_attention = attention_data['cls_attention']
            if hasattr(cls_attention, 'cpu'):
                cls_attention = cls_attention.cpu().numpy()

            axes[0, 0].plot(cls_attention, marker='o', markersize=2)
            axes[0, 0].set_title('CLS Token Attention Pattern')
            axes[0, 0].set_xlabel('Token Position')
            axes[0, 0].set_ylabel('Attention Weight')
            axes[0, 0].grid(True)

            # Attention heatmap (if 2D)
            if len(cls_attention.shape) > 1:
                im = axes[0, 1].imshow(cls_attention, cmap='viridis', aspect='auto')
                axes[0, 1].set_title('Attention Heatmap')
                axes[0, 1].set_xlabel('Token Position')
                axes[0, 1].set_ylabel('Layer/Head')
                plt.colorbar(im, ax=axes[0, 1])
            else:
                # 1D attention - show top tokens
                top_indices = np.argsort(cls_attention)[-20:]
                axes[0, 1].barh(range(len(top_indices)), cls_attention[top_indices])
                axes[0, 1].set_title('Top 20 Attention Weights')
                axes[0, 1].set_xlabel('Attention Weight')
                axes[0, 1].set_ylabel('Token Rank')

            # Attention distribution
            axes[1, 0].hist(cls_attention.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(stats.get('mean_attention', 0), color='red', linestyle='--',
                              label=f"Mean: {stats.get('mean_attention', 0):.3f}")
            axes[1, 0].axvline(stats.get('max_attention', 0), color='orange', linestyle='--',
                              label=f"Max: {stats.get('max_attention', 0):.3f}")
            axes[1, 0].set_title('Attention Distribution')
            axes[1, 0].set_xlabel('Attention Weight')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Statistics summary
        stats_text = (
            f"Max Attention: {stats.get('max_attention', 0):.3f}\n"
            f"Mean Attention: {stats.get('mean_attention', 0):.3f}\n"
            f"Attention Std: {stats.get('attention_std', 0):.3f}\n"
            f"Attention Entropy: {stats.get('attention_entropy', 0):.3f}\n"
            f"Sparsity: {stats.get('sparsity', 0):.3f}\n"
            f"Top 10% Concentration: {stats.get('concentration_top10', 0):.3f}\n"
            f"Attention Range: {stats.get('attention_range', 0):.3f}"
        )
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Attention Statistics')
        axes[1, 1].axis('off')

        plt.suptitle(f"Attention Analysis - Subject: {explanation['subject_id']}\n"
                    f"Prediction: Class {explanation['prediction']['class']} "
                    f"(Confidence: {explanation['prediction']['confidence']:.3f})",
                    fontsize=14, y=0.98)

        plt.tight_layout()

        # Save as bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        buffer.close()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, image_bytes

    def visualize_feature_attribution(self, explanation: Dict[str, Any],
                                    save_path: Optional[Path] = None) -> Tuple[plt.Figure, bytes]:
        """Visualize Core15+ feature attribution explanation"""
        attributions = explanation['attributions']['feature_attributions']
        feature_names = explanation['attributions']['feature_names']
        feature_values = explanation['attributions']['feature_values']
        importance = explanation['attributions']['feature_importance']

        n_methods = len(attributions)
        fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 12))

        if n_methods == 1:
            axes = axes.reshape(2, 1)

        for i, (method, attr_values) in enumerate(attributions.items()):
            # Top feature attributions
            if method in importance:
                top_features = importance[method].get('top_features', [])[:10]

                if top_features:
                    names = [f['feature_name'] for f in top_features]
                    values = [f['attribution_value'] for f in top_features]
                    colors_list = ['red' if v < 0 else 'blue' for v in values]

                    bars = axes[0, i].barh(range(len(values)), values, color=colors_list)
                    axes[0, i].set_yticks(range(len(values)))
                    axes[0, i].set_yticklabels(names, fontsize=8)
                    axes[0, i].set_title(f'{method.title()} - Top Features')
                    axes[0, i].set_xlabel('Attribution Value')
                    axes[0, i].grid(True, alpha=0.3)

                    # Add value labels on bars
                    for j, (bar, val) in enumerate(zip(bars, values)):
                        axes[0, i].text(val + 0.01 * max(abs(min(values)), max(values)),
                                       j, f'{val:.3f}', va='center', fontsize=8)

            # Attribution distribution
            if isinstance(attr_values, np.ndarray):
                attr_flat = attr_values.flatten()
            else:
                attr_flat = np.array(attr_values)

            axes[1, i].hist(attr_flat, bins=30, alpha=0.7, edgecolor='black')
            axes[1, i].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[1, i].set_title(f'{method.title()} - Distribution')
            axes[1, i].set_xlabel('Attribution Value')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f"Mean: {attr_flat.mean():.3f}\nStd: {attr_flat.std():.3f}"
            axes[1, i].text(0.7, 0.8, stats_text, transform=axes[1, i].transAxes,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.suptitle(f"Feature Attribution Analysis - Subject: {explanation['subject_id']}\n"
                    f"Prediction: Class {explanation['prediction']['class']} "
                    f"(Confidence: {explanation['prediction']['confidence']:.3f})",
                    fontsize=14, y=0.98)

        plt.tight_layout()

        # Save as bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        buffer.close()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, image_bytes

    def create_interactive_dashboard(self, explanation: Dict[str, Any]) -> str:
        """Create interactive Plotly dashboard"""
        # This would create an interactive HTML dashboard
        # For now, return a placeholder
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Explainability Dashboard - {explanation['subject_id']}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Model Explanation Dashboard</h1>
            <h2>Subject: {explanation['subject_id']}</h2>
            <p>Prediction: Class {explanation['prediction']['class']}
               (Confidence: {explanation['prediction']['confidence']:.3f})</p>

            <div id="explanation-plot" style="width:100%;height:600px;"></div>

            <script>
                // Interactive plots would be generated here using Plotly
                var data = [{
                    x: [1, 2, 3, 4, 5],
                    y: [1, 4, 2, 3, 5],
                    type: 'scatter'
                }];

                Plotly.newPlot('explanation-plot', data);
            </script>
        </body>
        </html>
        """
        return dashboard_html

    def create_comparison_visualization(self, explanations: List[Dict[str, Any]],
                                      save_path: Optional[Path] = None) -> Tuple[plt.Figure, bytes]:
        """Create comparison visualization across multiple explanations"""
        n_explanations = len(explanations)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confidence comparison
        subjects = [exp['subject_id'] for exp in explanations]
        confidences = [exp['prediction']['confidence'] for exp in explanations]
        predicted_classes = [exp['prediction']['class'] for exp in explanations]

        axes[0, 0].bar(range(len(subjects)), confidences,
                      color=['blue' if c == 0 else 'red' for c in predicted_classes])
        axes[0, 0].set_title('Prediction Confidence Comparison')
        axes[0, 0].set_xlabel('Subject')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].set_xticks(range(len(subjects)))
        axes[0, 0].set_xticklabels(subjects, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Class distribution
        class_counts = pd.Series(predicted_classes).value_counts()
        axes[0, 1].pie(class_counts.values, labels=[f'Class {i}' for i in class_counts.index],
                      autopct='%1.1f%%')
        axes[0, 1].set_title('Predicted Class Distribution')

        # Attribution comparison (if available)
        if all('attributions' in exp for exp in explanations):
            # This would create more sophisticated attribution comparisons
            axes[1, 0].text(0.5, 0.5, 'Attribution Comparison\n(Feature implementation needed)',
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 0].set_title('Cross-Subject Attribution Patterns')

        # Performance metrics summary
        metrics_text = f"""
        Total Subjects: {n_explanations}
        Average Confidence: {np.mean(confidences):.3f}
        Confidence Std: {np.std(confidences):.3f}
        Class 0 Count: {sum(1 for c in predicted_classes if c == 0)}
        Class 1 Count: {sum(1 for c in predicted_classes if c == 1)}
        """
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')

        plt.suptitle('Multi-Subject Explanation Comparison', fontsize=16)
        plt.tight_layout()

        # Save as bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        buffer.close()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, image_bytes

    def visualize_explanation(self, explanation: Dict[str, Any],
                            save_path: Optional[Path] = None) -> Tuple[plt.Figure, bytes]:
        """Auto-detect explanation type and create appropriate visualization"""
        explanation_type = explanation.get('metadata', {}).get('explanation_type', 'unknown')

        if 'attribution' in explanation and 'cam' in explanation['attribution']:
            return self.visualize_grad_cam(explanation, save_path)
        elif 'attention' in explanation:
            return self.visualize_attention(explanation, save_path)
        elif 'attributions' in explanation and 'feature_attributions' in explanation['attributions']:
            return self.visualize_feature_attribution(explanation, save_path)
        else:
            # Create generic visualization
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f'Explanation visualization for {explanation_type}\nnot implemented',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax.set_title(f"Generic Explanation - Subject: {explanation.get('subject_id', 'Unknown')}")
            ax.axis('off')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_bytes = buffer.getvalue()
            buffer.close()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig, image_bytes