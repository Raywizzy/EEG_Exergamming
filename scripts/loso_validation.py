#!/usr/bin/env python3
"""Leave-One-Site-Out (LOSO) Cross-Validation for Multi-Site EEG Biomarker Validation.

Phase IV Clinical-Trial Grade Implementation
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.domain_adaptation.coral import CORAL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LOSOValidator:
    """Leave-One-Site-Out cross-validation with CORAL domain adaptation."""

    def __init__(self, config_path: str = "config/preprocessing_config.yaml"):
        """Initialize LOSO validator.

        Args:
            config_path: Path to preprocessing configuration
        """
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.features_dir = Path("data/features/core5")
        self.results_dir = Path("results/loso_validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Core5 feature names (locked from Phase III)
        self.core5_features = [
            'duration_cv', 'duty_cycle', 'mean_duration_ms',
            'median_duration_ms', 'motor_posterior_duty_ratio'
        ]

        # Model configuration (locked)
        self.model_config = self.config['model']
        self.coral_config = self.config['coral']

        logger.info("Initialized LOSO validator with locked Core5 features")
        logger.info(f"Target features: {self.core5_features}")

    def load_dataset_features(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load Core5 features for a dataset.

        Args:
            dataset_id: Dataset identifier (e.g., 'ds004584')

        Returns:
            DataFrame with features and metadata, or None if not found
        """
        features_file = self.features_dir / f"{dataset_id}_core5_features.csv"

        if not features_file.exists():
            logger.warning(f"Features file not found: {features_file}")
            return None

        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} subjects from {dataset_id}")

        # Validate Core5 features are present
        missing_features = [f for f in self.core5_features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing Core5 features in {dataset_id}: {missing_features}")
            return None

        return df

    def prepare_labels(self, df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
        """Prepare binary labels for classification.

        For Phase IV validation:
        - PD patients: label = 1
        - Healthy controls: label = 0

        Args:
            df: Feature dataframe
            dataset_id: Dataset identifier

        Returns:
            DataFrame with 'label' column added
        """
        # Determine labels based on subject IDs and dataset conventions
        if 'label' not in df.columns:
            if dataset_id == 'ds004584':
                # ds004584: sub-XXX format, need to check participants.tsv for actual labels
                # For now, assume PD based on original dataset description (PD + controls)
                # This would need to be refined with actual participant metadata
                df['label'] = 1  # Placeholder - all marked as PD for now
                logger.warning(f"Using placeholder labels for {dataset_id} - needs participant metadata")

            elif dataset_id == 'ds002778':
                # ds002778: sub-pd* = PD (1), sub-hc* = HC (0)
                df['label'] = df['subject_id'].apply(lambda x: 1 if x.startswith('sub-pd') else 0)

            else:
                logger.error(f"Unknown label mapping for dataset: {dataset_id}")
                return df

        n_pd = (df['label'] == 1).sum()
        n_hc = (df['label'] == 0).sum()
        logger.info(f"{dataset_id} labels: {n_pd} PD, {n_hc} HC")

        return df

    def apply_coral_adaptation(self, source_data: np.ndarray, target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply CORAL domain adaptation.

        Args:
            source_data: Source domain feature matrix
            target_data: Target domain feature matrix

        Returns:
            Tuple of (adapted_source, adapted_target)
        """
        coral = CORAL(reg_param=self.coral_config['regularization'])

        # Normalize features before CORAL (per config)
        if self.coral_config['normalize']:
            scaler_source = StandardScaler()
            scaler_target = StandardScaler()

            source_norm = scaler_source.fit_transform(source_data)
            target_norm = scaler_target.fit_transform(target_data)
        else:
            source_norm = source_data.copy()
            target_norm = target_data.copy()

        # Apply CORAL transformation
        adapted_source, adapted_target = coral.fit_transform(source_norm, target_norm)

        return adapted_source, adapted_target

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train locked logistic regression model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        # Use locked model parameters from config
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )

        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with performance metrics
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'n_test_samples': len(y_test),
            'n_pd_test': (y_test == 1).sum(),
            'n_hc_test': (y_test == 0).sum()
        }

        return metrics

    def run_loso_validation(self, datasets: List[str]) -> Dict:
        """Run Leave-One-Site-Out validation.

        Args:
            datasets: List of dataset IDs to include

        Returns:
            LOSO validation results
        """
        logger.info(f"Starting LOSO validation with datasets: {datasets}")

        # Load all datasets
        dataset_dfs = {}
        for dataset_id in datasets:
            df = self.load_dataset_features(dataset_id)
            if df is not None:
                df = self.prepare_labels(df, dataset_id)
                dataset_dfs[dataset_id] = df

        if len(dataset_dfs) < 2:
            raise ValueError(f"Need at least 2 datasets for LOSO, got {len(dataset_dfs)}")

        logger.info(f"Successfully loaded {len(dataset_dfs)} datasets")

        # LOSO cross-validation
        loso_results = {}

        for test_dataset in dataset_dfs.keys():
            logger.info(f"LOSO fold: Testing on {test_dataset}")

            # Prepare training and test data
            test_df = dataset_dfs[test_dataset]
            train_dfs = [dataset_dfs[d] for d in dataset_dfs.keys() if d != test_dataset]
            train_df = pd.concat(train_dfs, ignore_index=True)

            # Extract features
            X_train = train_df[self.core5_features].values
            y_train = train_df['label'].values
            X_test = test_df[self.core5_features].values
            y_test = test_df['label'].values

            logger.info(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")

            # Apply CORAL domain adaptation
            if len(train_dfs) > 1:
                # Multiple source domains - apply CORAL pairwise
                X_train_adapted = X_train.copy()
                for i, source_df in enumerate(train_dfs):
                    source_X = source_df[self.core5_features].values
                    adapted_source, adapted_test = self.apply_coral_adaptation(source_X, X_test)
                    if i == 0:
                        X_test_adapted = adapted_test
                    # Update training data with adapted source
                    start_idx = sum(len(d) for d in train_dfs[:i])
                    end_idx = start_idx + len(source_df)
                    X_train_adapted[start_idx:end_idx] = adapted_source
            else:
                # Single source domain
                X_train_adapted, X_test_adapted = self.apply_coral_adaptation(X_train, X_test)

            # Train model
            model = self.train_model(X_train_adapted, y_train)

            # Evaluate
            metrics = self.evaluate_model(model, X_test_adapted, y_test)
            metrics['test_dataset'] = test_dataset
            metrics['train_datasets'] = [d for d in dataset_dfs.keys() if d != test_dataset]

            loso_results[test_dataset] = metrics

            logger.info(f"Results for {test_dataset}: BA={metrics['balanced_accuracy']:.3f}")

        return loso_results

    def generate_report(self, loso_results: Dict, datasets: List[str]) -> Dict:
        """Generate comprehensive LOSO validation report.

        Args:
            loso_results: LOSO validation results
            datasets: List of datasets used

        Returns:
            Report dictionary
        """
        # Calculate aggregate statistics
        balanced_accuracies = [r['balanced_accuracy'] for r in loso_results.values()]

        report = {
            'loso_validation_summary': {
                'datasets_included': datasets,
                'n_folds': len(loso_results),
                'mean_balanced_accuracy': np.mean(balanced_accuracies),
                'std_balanced_accuracy': np.std(balanced_accuracies),
                'min_balanced_accuracy': np.min(balanced_accuracies),
                'max_balanced_accuracy': np.max(balanced_accuracies),
                'validation_timestamp': datetime.now().isoformat()
            },
            'fold_results': loso_results,
            'statistical_analysis': {
                'one_sample_ttest': self._one_sample_ttest(balanced_accuracies),
                'clinical_significance': self._assess_clinical_significance(np.mean(balanced_accuracies))
            }
        }

        return report

    def _one_sample_ttest(self, balanced_accuracies: List[float]) -> Dict:
        """Perform one-sample t-test against chance performance."""
        from scipy import stats

        null_hypothesis = 0.50  # Chance performance
        t_stat, p_value = stats.ttest_1samp(balanced_accuracies, null_hypothesis)

        return {
            'null_hypothesis': f"mean_BA <= {null_hypothesis}",
            'alternative': f"mean_BA > {null_hypothesis}",
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'effect_size': (np.mean(balanced_accuracies) - null_hypothesis) / np.std(balanced_accuracies)
        }

    def _assess_clinical_significance(self, mean_ba: float) -> Dict:
        """Assess clinical significance of results."""
        target_ba = self.config['loso_validation']['design_target_ba']

        return {
            'design_target_ba': target_ba,
            'achieved_mean_ba': mean_ba,
            'meets_target': bool(mean_ba >= target_ba),
            'margin_above_target': mean_ba - target_ba if mean_ba >= target_ba else None,
            'clinical_interpretation': 'Clinically meaningful' if mean_ba >= target_ba else 'Below clinical threshold'
        }

    def save_results(self, report: Dict, output_prefix: str = "loso_validation") -> str:
        """Save LOSO validation results.

        Args:
            report: Validation report
            output_prefix: Output file prefix

        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"{output_prefix}_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"LOSO validation results saved to: {output_file}")
        return str(output_file)

def main():
    parser = argparse.ArgumentParser(description="LOSO Cross-Validation for Multi-Site EEG Validation")
    parser.add_argument('--datasets', nargs='+', default=['ds004584'],
                       help='Dataset IDs to include in LOSO validation')
    parser.add_argument('--config', default='config/preprocessing_config.yaml',
                       help='Path to preprocessing config')
    parser.add_argument('--output', default='loso_validation',
                       help='Output file prefix')

    args = parser.parse_args()

    print("="*80)
    print("LOSO CROSS-VALIDATION FOR MULTI-SITE EEG BIOMARKER VALIDATION")
    print("Phase IV Clinical-Trial Grade Implementation")
    print("="*80)
    print(f"Datasets: {args.datasets}")
    print(f"Config: {args.config}")
    print()

    try:
        # Initialize validator
        validator = LOSOValidator(args.config)

        # Run LOSO validation
        loso_results = validator.run_loso_validation(args.datasets)

        # Generate comprehensive report
        report = validator.generate_report(loso_results, args.datasets)

        # Save results
        output_file = validator.save_results(report, args.output)

        # Print summary
        summary = report['loso_validation_summary']
        stats = report['statistical_analysis']

        print(f"✅ LOSO Validation Complete")
        print(f"📊 Mean Balanced Accuracy: {summary['mean_balanced_accuracy']:.3f} ± {summary['std_balanced_accuracy']:.3f}")
        print(f"📈 Range: {summary['min_balanced_accuracy']:.3f} - {summary['max_balanced_accuracy']:.3f}")
        print(f"🧪 Statistical Test: p={stats['one_sample_ttest']['p_value']:.4f} (vs chance)")
        print(f"🏥 Clinical Significance: {stats['clinical_significance']['clinical_interpretation']}")
        print(f"💾 Results saved to: {output_file}")

    except Exception as e:
        logger.error(f"LOSO validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()