#!/usr/bin/env python3
"""
Full Dataset Processing Pipeline for ds004584
Clinical-Trial Grade: Preprocessing + Core5 Feature Extraction + QC

Usage:
    python scripts/process_full_dataset.py --dataset ds004584 --max-subjects 10
    python scripts/process_full_dataset.py --dataset ds004584  # Process all
"""

import sys
import os
from pathlib import Path
import argparse
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import mne
from scipy import signal
from scipy.signal import hilbert
from scipy import stats

# Add src to path
sys.path.append('src')
from preprocess.pipeline import EEGPreprocessor
from features.core5 import Core5FeatureExtractor, BetaBurstDetector

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullDatasetProcessor:
    """Process full dataset: preprocessing + feature extraction + QC."""

    def __init__(self, config_path: str, dataset_id: str):
        """Initialize processor.

        Args:
            config_path: Path to preprocessing config
            dataset_id: Dataset identifier
        """
        self.config_path = config_path
        self.dataset_id = dataset_id

        # Initialize components
        self.preprocessor = EEGPreprocessor(config_path)
        self.feature_extractor = Core5FeatureExtractor(config_path)

        # Output directories
        self.base_dir = Path("data")
        self.interim_dir = self.base_dir / "interim" / dataset_id
        self.features_dir = self.base_dir / "features" / "core5"
        self.qc_dir = Path("results") / "qc" / dataset_id

        # Create directories
        for dir_path in [self.interim_dir, self.features_dir, self.qc_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized processor for dataset: {dataset_id}")

    def process_single_subject(self, raw: mne.io.Raw, metadata: Dict) -> Optional[Dict]:
        """Process single subject through full pipeline.

        Args:
            raw: Raw EEG data
            metadata: Subject metadata

        Returns:
            Dictionary with processing results or None if failed
        """
        subject_id = metadata.get('subject_id', 'unknown')
        start_time = datetime.now()

        logger.info(f"Processing {subject_id}")

        try:
            # Preprocessing
            epochs, beta_data, qc_metrics = self.preprocessor.preprocess_subject(raw, metadata)

            if epochs is None:
                logger.warning(f"Preprocessing failed for {subject_id}")
                return {
                    'subject_id': subject_id,
                    'success': False,
                    'stage': 'preprocessing',
                    'qc_metrics': qc_metrics
                }

            # Core5 feature extraction
            features = self.feature_extractor.extract_core5_features(
                beta_data, epochs.info['sfreq'], epochs.ch_names, metadata
            )

            # Save intermediate results
            subject_results = {
                'epochs': epochs,
                'beta_data': beta_data,
                'features': features,
                'qc_metrics': qc_metrics,
                'metadata': metadata
            }

            # Save to disk
            results_file = self.interim_dir / f"{subject_id}_processed.pkl"
            with open(results_file, 'wb') as f:
                # Save without MNE objects for space efficiency
                save_data = {
                    'features': features,
                    'qc_metrics': qc_metrics,
                    'metadata': metadata,
                    'epochs_info': {
                        'n_epochs': len(epochs),
                        'sfreq': epochs.info['sfreq'],
                        'ch_names': epochs.ch_names
                    }
                }
                pickle.dump(save_data, f)

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ {subject_id}: {len(epochs)} epochs, {features['n_bursts_total']} bursts, {processing_time:.1f}s")

            return {
                'subject_id': subject_id,
                'success': True,
                'stage': 'complete',
                'features': features,
                'qc_metrics': qc_metrics,
                'processing_time_s': processing_time
            }

        except Exception as e:
            logger.error(f"❌ {subject_id}: {e}")
            return {
                'subject_id': subject_id,
                'success': False,
                'stage': 'error',
                'error': str(e),
                'processing_time_s': (datetime.now() - start_time).total_seconds()
            }

    def process_dataset(self, max_subjects: Optional[int] = None) -> Dict:
        """Process entire dataset.

        Args:
            max_subjects: Maximum number of subjects to process (for testing)

        Returns:
            Processing summary
        """
        start_time = datetime.now()
        logger.info(f"Starting full dataset processing: {self.dataset_id}")

        # Update artifact rejection settings for this dataset
        self.preprocessor._update_artifact_reject_for_dataset(self.dataset_id)

        # Load raw files from preprocessor
        dataset_config = self.preprocessor.config['datasets'][self.dataset_id]

        if dataset_config['type'] == 'BIDS':
            bids_path = Path(dataset_config['root'])
        else:
            bids_path = Path("data/bids") / self.dataset_id

        raw_files = self.preprocessor._load_bids_data(bids_path, self.dataset_id)

        if not raw_files:
            raise ValueError(f"No EEG files found for {self.dataset_id}")

        # Limit subjects for testing
        if max_subjects:
            raw_files = raw_files[:max_subjects]
            logger.info(f"Processing first {len(raw_files)} subjects (limit: {max_subjects})")

        logger.info(f"Found {len(raw_files)} subjects to process")

        # Process subjects
        results = []
        features_list = []
        qc_list = []

        for i, (raw, metadata) in enumerate(raw_files, 1):
            logger.info(f"Subject {i}/{len(raw_files)}: {metadata['subject_id']}")

            result = self.process_single_subject(raw, metadata)
            results.append(result)

            if result['success']:
                features_list.append(result['features'])
                qc_list.append(result['qc_metrics'])

        # Create summary statistics
        successful_subjects = [r for r in results if r['success']]
        failed_subjects = [r for r in results if not r['success']]

        processing_summary = {
            'dataset_id': self.dataset_id,
            'processing_timestamp': datetime.now().isoformat(),
            'total_subjects': len(raw_files),
            'successful_subjects': len(successful_subjects),
            'failed_subjects': len(failed_subjects),
            'success_rate': len(successful_subjects) / len(raw_files) if raw_files else 0,
            'total_processing_time_s': (datetime.now() - start_time).total_seconds()
        }

        # Save features CSV
        if features_list:
            features_df = pd.DataFrame(features_list)
            features_file = self.features_dir / f"{self.dataset_id}_core5_features.csv"
            features_df.to_csv(features_file, index=False)
            logger.info(f"✅ Saved {len(features_df)} Core5 feature vectors to {features_file}")

            # Feature validation
            validation_report = self.feature_extractor.validate_features(features_df)
            processing_summary['feature_validation'] = validation_report

        # Save QC report
        if qc_list:
            qc_df = pd.DataFrame(qc_list)
            qc_file = self.qc_dir / f"{self.dataset_id}_full_qc_report.csv"
            qc_df.to_csv(qc_file, index=False)
            logger.info(f"✅ Saved QC report to {qc_file}")

            # QC statistics
            if len(successful_subjects) > 0:
                qc_stats = {
                    'mean_epochs_per_subject': float(qc_df[qc_df['success']]['n_epochs'].mean()),
                    'std_epochs_per_subject': float(qc_df[qc_df['success']]['n_epochs'].std()),
                    'mean_artifact_proportion': float(qc_df[qc_df['success']]['artifact_proportion'].mean()),
                    'std_artifact_proportion': float(qc_df[qc_df['success']]['artifact_proportion'].std()),
                    'mean_processing_time_s': float(qc_df[qc_df['success']]['processing_time_s'].mean())
                }
                processing_summary['qc_statistics'] = qc_stats

        # Save processing summary
        summary_file = self.qc_dir / f"{self.dataset_id}_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(processing_summary, f, indent=2)

        logger.info(f"✅ Processing complete: {processing_summary['success_rate']:.1%} success rate")
        logger.info(f"✅ Summary saved to {summary_file}")

        return processing_summary

    def generate_qc_dashboard(self) -> None:
        """Generate comprehensive QC dashboard."""
        logger.info("Generating QC dashboard...")

        # Load QC data
        qc_file = self.qc_dir / f"{self.dataset_id}_full_qc_report.csv"
        features_file = self.features_dir / f"{self.dataset_id}_core5_features.csv"

        if not qc_file.exists():
            logger.warning("QC file not found - run processing first")
            return

        qc_df = pd.read_csv(qc_file)
        successful = qc_df[qc_df['success'] == True]

        # Generate plots
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        if len(successful) > 0:
            # 1. Epochs distribution
            axes[0, 0].hist(successful['n_epochs'], bins=20, alpha=0.7, color='skyblue')
            axes[0, 0].set_xlabel('Number of Epochs')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Epochs per Subject Distribution')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Artifact proportion
            axes[0, 1].hist(successful['artifact_proportion'] * 100, bins=20, alpha=0.7, color='lightcoral')
            axes[0, 1].set_xlabel('Artifact Proportion (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Artifact Distribution')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Processing time
            axes[0, 2].hist(successful['processing_time_s'], bins=20, alpha=0.7, color='lightgreen')
            axes[0, 2].set_xlabel('Processing Time (s)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Processing Time Distribution')
            axes[0, 2].grid(True, alpha=0.3)

            # 4. Signal quality
            if 'signal_quality_score' in successful.columns:
                axes[1, 0].hist(successful['signal_quality_score'], bins=20, alpha=0.7, color='gold')
                axes[1, 0].set_xlabel('Signal Quality Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Signal Quality Distribution')
                axes[1, 0].grid(True, alpha=0.3)

        # 5. Success/failure breakdown
        n_success = len(qc_df[qc_df['success'] == True])
        n_failed = len(qc_df[qc_df['success'] == False])
        success_rate = n_success / len(qc_df) if len(qc_df) > 0 else 0

        # Success/failure pie chart
        if n_failed > 0:
            axes[1, 1].pie([n_success, n_failed], labels=['Success', 'Failed'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            axes[1, 1].set_title(f'Processing Success Rate ({success_rate:.1%})')
        else:
            axes[1, 1].text(0.5, 0.5, f'100% Success\n({n_success} subjects)', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Processing Success Rate')

        # 6. Feature correlation if available
        if features_file.exists():
            features_df = pd.read_csv(features_file)
            core5_cols = ['duration_cv', 'duty_cycle', 'mean_duration_ms', 'median_duration_ms', 'motor_posterior_duty_ratio']
            available_cols = [col for col in core5_cols if col in features_df.columns]

            if len(available_cols) > 1:
                corr_matrix = features_df[available_cols].corr()
                sns.heatmap(corr_matrix, annot=True, ax=axes[1, 2], cmap='coolwarm', center=0)
                axes[1, 2].set_title('Core5 Feature Correlations')
            else:
                axes[1, 2].text(0.5, 0.5, 'Features not available', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Core5 Feature Correlations')
        else:
            axes[1, 2].text(0.5, 0.5, 'Features not available', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Core5 Feature Correlations')

        plt.tight_layout()

        # Save dashboard
        dashboard_file = self.qc_dir / f"{self.dataset_id}_qc_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ QC dashboard saved to {dashboard_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Process full dataset with preprocessing and feature extraction")
    parser.add_argument('--dataset', required=True, help='Dataset ID to process')
    parser.add_argument('--config', default='config/preprocessing_config.yaml', help='Config file path')
    parser.add_argument('--max-subjects', type=int, help='Maximum subjects to process (for testing)')
    parser.add_argument('--dashboard-only', action='store_true', help='Only generate QC dashboard')

    args = parser.parse_args()

    print("=" * 80)
    print("FULL DATASET PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Max subjects: {args.max_subjects or 'All'}")
    print(f"Dashboard only: {args.dashboard_only}")
    print()

    # Initialize processor
    processor = FullDatasetProcessor(args.config, args.dataset)

    if args.dashboard_only:
        # Generate dashboard only
        processor.generate_qc_dashboard()
    else:
        # Full processing
        summary = processor.process_dataset(args.max_subjects)

        # Print summary
        print("=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total subjects: {summary['total_subjects']}")
        print(f"Successful: {summary['successful_subjects']}")
        print(f"Failed: {summary['failed_subjects']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total time: {summary['total_processing_time_s']:.1f} seconds")

        if 'qc_statistics' in summary:
            qc = summary['qc_statistics']
            print(f"Epochs per subject: {qc['mean_epochs_per_subject']:.1f} ± {qc['std_epochs_per_subject']:.1f}")
            print(f"Artifact proportion: {qc['mean_artifact_proportion']:.1%} ± {qc['std_artifact_proportion']:.1%}")
            print(f"Processing time: {qc['mean_processing_time_s']:.1f}s per subject")

        if 'feature_validation' in summary:
            fv = summary['feature_validation']
            print(f"Features extracted: {fv['n_subjects']} subjects")

        print("=" * 80)

        # Generate QC dashboard
        processor.generate_qc_dashboard()


if __name__ == "__main__":
    main()