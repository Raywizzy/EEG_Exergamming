#!/usr/bin/env python3
"""
Automated manuscript updater
Syncs manuscript with latest validation results and badges
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManuscriptUpdater:
    """Update manuscript with latest validation results."""

    def __init__(self):
        self.manuscript_path = Path("manuscripts/PhaseIV_MultiSite_Validation.md")
        self.results_dir = Path("results")
        self.badges_dir = Path("results/badges")

    def get_latest_results(self):
        """Extract latest validation results from summary files."""

        # Try to get LOSO summary
        loso_summary_files = list(self.results_dir.glob("**/loso_overall_summary.csv"))
        if loso_summary_files:
            latest_summary = max(loso_summary_files, key=lambda x: x.stat().st_mtime)
            summary_df = pd.read_csv(latest_summary)

            results = {}
            for _, row in summary_df.iterrows():
                metric = row['Metric']
                value = row['Value']

                if 'Mean Balanced Accuracy' in metric:
                    results['mean_ba'] = value
                elif 'Total Subjects' in metric:
                    results['n_total'] = value
                elif 'Total PD' in metric:
                    results['n_pd'] = value
                elif 'Total Healthy' in metric:
                    results['n_hc'] = value
                elif 'Number of Sites' in metric:
                    results['n_sites'] = value
                elif 'Range' in metric:
                    results['ba_range'] = value
                elif 'Folds Above Chance' in metric:
                    results['above_chance'] = value
                elif 'Clinical Target' in metric:
                    results['clinical_met'] = value

            return results

        return None

    def get_dataset_info(self):
        """Get individual dataset information."""
        datasets = {}

        # Check for feature files to get dataset info
        feature_files = list(Path("data/features/core5").glob("*_core5_features.csv"))

        for file in feature_files:
            dataset_id = file.stem.replace("_core5_features", "")
            try:
                df = pd.read_csv(file)
                datasets[dataset_id] = {
                    'n_total': len(df),
                    'n_pd': len(df[df['label'] == 1]) if 'label' in df.columns else 'TBD',
                    'n_hc': len(df[df['label'] == 0]) if 'label' in df.columns else 'TBD'
                }
            except Exception as e:
                logger.warning(f"Could not process {file}: {e}")

        return datasets

    def update_abstract_results(self, content, results):
        """Update the results section in abstract."""
        if not results:
            return content

        # Find and update abstract results
        abstract_pattern = r'(\*\*Results\*\*:.*?)(\*\*\[Three-site results pending\]\*\*)'

        if results.get('mean_ba'):
            new_results = f"""**Results**: ![LOSO Overall Performance](../results/badges/loso_overall_performance.svg) ![Clinical Significance](../results/badges/loso_clinical_significance.svg)

**Two-site validation achieved**: Mean balanced accuracy {results['mean_ba']} (range: {results.get('ba_range', 'TBD')}). {results.get('above_chance', 'TBD')} folds above chance, {results.get('clinical_met', 'TBD')} meeting clinical threshold. CORAL domain adaptation successfully harmonized cross-site differences.

**[Three-site results pending]**"""

            content = re.sub(abstract_pattern, new_results + r'\2', content, flags=re.DOTALL)

        return content

    def update_dataset_table(self, content, datasets):
        """Update dataset characteristics table."""
        if not datasets:
            return content

        # Build table rows
        table_rows = []
        total_subjects = 0
        total_pd = 0
        total_hc = 0

        for dataset_id, info in datasets.items():
            if isinstance(info['n_total'], int):
                total_subjects += info['n_total']
            if isinstance(info['n_pd'], int):
                total_pd += info['n_pd']
            if isinstance(info['n_hc'], int):
                total_hc += info['n_hc']

            success_rate = "TBD"  # Could be calculated from QC files

            table_rows.append(f"| {dataset_id} | {info['n_total']} | {info['n_pd']} | {info['n_hc']} | [TBD] | [TBD] | {success_rate} |")

        # Add total row
        table_rows.append(f"| **Total** | **{total_subjects}** | **{total_pd}** | **{total_hc}** | **[TBD]** | **[TBD]** | **[TBD]** |")

        # Replace table
        table_pattern = r'(\| Site \| N Total.*?\n)(.*?)(\| \*\*Total\*\* \|.*?\n)'
        new_table = "\n".join(table_rows)

        content = re.sub(table_pattern, r'\1' + new_table, content, flags=re.DOTALL)

        return content

    def update_loso_results(self, content, results):
        """Update LOSO results section."""
        if not results:
            return content

        # Update summary statistics
        if results.get('mean_ba'):
            summary_pattern = r'(\*\*Summary Statistics\*\*:\n- \*\*Mean Balanced Accuracy\*\*: )([^±\n]+)(.*?)(\*\*Overall Clinical Status\*\*:)'

            replacement = f"""**Summary Statistics**:
- **Mean Balanced Accuracy**: {results['mean_ba']} ± [TBD]%
- **Range**: {results.get('ba_range', 'TBD')}
- **Folds Above Chance**: {results.get('above_chance', 'TBD')}
- **Folds Meeting Clinical Target**: {results.get('clinical_met', 'TBD')}
- **Overall Clinical Status**:"""

            content = re.sub(summary_pattern, replacement, content, flags=re.DOTALL)

        return content

    def update_timestamp(self, content):
        """Update last modified timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update auto-update status
        timestamp_pattern = r'(\*\*Last Updated\*\*: ).*'
        content = re.sub(timestamp_pattern, f"**Last Updated**: {timestamp}", content)

        return content

    def update_manuscript(self):
        """Main update function."""
        if not self.manuscript_path.exists():
            logger.error(f"Manuscript not found: {self.manuscript_path}")
            return False

        logger.info("Updating manuscript with latest validation results...")

        # Read current manuscript
        with open(self.manuscript_path, 'r') as f:
            content = f.read()

        # Get latest results
        results = self.get_latest_results()
        datasets = self.get_dataset_info()

        if results:
            logger.info(f"Found results: {results}")

        if datasets:
            logger.info(f"Found datasets: {list(datasets.keys())}")

        # Update sections
        content = self.update_abstract_results(content, results)
        content = self.update_dataset_table(content, datasets)
        content = self.update_loso_results(content, results)
        content = self.update_timestamp(content)

        # Write updated manuscript
        with open(self.manuscript_path, 'w') as f:
            f.write(content)

        logger.info(f"✅ Manuscript updated: {self.manuscript_path}")
        return True

def main():
    """Main execution function."""
    updater = ManuscriptUpdater()
    success = updater.update_manuscript()

    if success:
        print("🎉 Manuscript updated successfully!")
        print(f"📄 Check: manuscripts/PhaseIV_MultiSite_Validation.md")
        print(f"🔄 Auto-synced with latest validation results")
    else:
        print("❌ Manuscript update failed")

if __name__ == "__main__":
    main()