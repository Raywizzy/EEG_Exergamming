#!/usr/bin/env python3
"""
Figure Manager for EEG Exergaming Project
Handles consistent figure saving with timestamps and metadata.

All figures must be saved with:
- Timestamped filenames
- Both PNG and SVG formats
- Tight bounding boxes
- Short descriptive captions
- Metadata logging
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from datetime import datetime, timezone
import json

# Set consistent style
plt.style.use('default')
sns.set_palette("husl")
matplotlib.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

class FigureManager:
    """Manages figure saving with consistent naming and metadata."""

    def __init__(self, base_path="results/figures"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create stage directories
        self.stages = {
            "raw": "01_raw",
            "preproc": "02_preproc",
            "epochs": "03_events_epochs",
            "features": "04_psd_features",
            "models": "05_models_cv",
            "importance": "06_importance_explain",
            "external": "07_external"
        }

        for stage_dir in self.stages.values():
            (self.base_path / stage_dir).mkdir(exist_ok=True)

    def save_figure(self, fig, filename, stage, caption="", metadata=None):
        """
        Save figure with consistent naming and metadata.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Base filename (without extension)
        stage : str
            Stage name (raw, preproc, epochs, features, models, importance, external)
        caption : str
            Short description for filename
        metadata : dict
            Additional metadata to log
        """
        if stage not in self.stages:
            raise ValueError(f"Stage must be one of: {list(self.stages.keys())}")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")

        # Create filename with timestamp and caption
        if caption:
            full_filename = f"{filename}_{caption}_{timestamp}"
        else:
            full_filename = f"{filename}_{timestamp}"

        # Clean filename (remove special characters)
        full_filename = "".join(c for c in full_filename if c.isalnum() or c in "._-")

        stage_path = self.base_path / self.stages[stage]

        # Save PNG and SVG
        png_path = stage_path / f"{full_filename}.png"
        svg_path = stage_path / f"{full_filename}.svg"

        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        fig.savefig(svg_path, format='svg', bbox_inches='tight')

        # Log metadata
        self._log_figure_metadata(
            stage_path / f"{full_filename}_metadata.json",
            {
                "filename": full_filename,
                "stage": stage,
                "caption": caption,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "png_path": str(png_path),
                "svg_path": str(svg_path),
                "metadata": metadata or {}
            }
        )

        print(f"✓ Figure saved: {png_path}")
        print(f"✓ Figure saved: {svg_path}")

        return png_path, svg_path

    def _log_figure_metadata(self, metadata_path, metadata):
        """Log figure metadata to JSON file."""
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def create_subplot_figure(self, nrows, ncols, figsize=None):
        """Create figure with subplots using consistent style."""
        if figsize is None:
            figsize = (4*ncols, 3*nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle("", y=0.98)  # Empty title, will be set by caller

        return fig, axes

    def close_all(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all')

# Global figure manager instance
figure_manager = FigureManager()

def save_figure(fig, filename, stage, caption="", metadata=None):
    """Convenience function for saving figures."""
    return figure_manager.save_figure(fig, filename, stage, caption, metadata)

def create_figure(nrows=1, ncols=1, figsize=None):
    """Convenience function for creating figures."""
    return figure_manager.create_subplot_figure(nrows, ncols, figsize)

def close_all_figures():
    """Convenience function for closing all figures."""
    figure_manager.close_all()

# Example usage functions for each stage
def save_raw_eeg_plot(fig, subject_id, caption=""):
    """Save raw EEG plot."""
    return save_figure(fig, f"raw_eeg_subj{subject_id}", "raw", caption)

def save_preprocessing_plot(fig, subject_id, process_type, caption=""):
    """Save preprocessing plot (filtering, ICA, etc.)."""
    return save_figure(fig, f"preproc_{process_type}_subj{subject_id}", "preproc", caption)

def save_epoch_plot(fig, subject_id, epoch_type, caption=""):
    """Save epoch-related plot."""
    return save_figure(fig, f"epochs_{epoch_type}_subj{subject_id}", "epochs", caption)

def save_feature_plot(fig, feature_type, caption=""):
    """Save feature extraction plot."""
    return save_figure(fig, f"features_{feature_type}", "features", caption)

def save_model_plot(fig, model_type, caption=""):
    """Save model performance plot."""
    return save_figure(fig, f"model_{model_type}", "models", caption)

def save_importance_plot(fig, analysis_type, caption=""):
    """Save feature importance/explainability plot."""
    return save_figure(fig, f"importance_{analysis_type}", "importance", caption)

def save_external_validation_plot(fig, analysis_type, caption=""):
    """Save external validation plot."""
    return save_figure(fig, f"external_{analysis_type}", "external", caption)