#!/usr/bin/env python3
"""
Generate dynamic SVG badges for LOSO validation results
Auto-updates based on BA ≥ 65% clinical threshold
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LOSOBadgeGenerator:
    """Generate dynamic performance badges for LOSO validation."""

    def __init__(self):
        self.results_dir = Path("results/loso_validation")
        self.badges_dir = Path("results/badges")
        self.badges_dir.mkdir(parents=True, exist_ok=True)

        # Clinical thresholds
        self.clinical_threshold = 0.65  # 65% BA for clinical significance
        self.chance_threshold = 0.50    # 50% chance performance

    def create_svg_badge(self, label: str, value: str, color: str, width: int = 120) -> str:
        """Create SVG badge with dynamic coloring.

        Args:
            label: Left side text (e.g., "LOSO BA")
            value: Right side text (e.g., "67.3%")
            color: Badge color ("green", "red", "orange", "blue")
            width: Total badge width

        Returns:
            SVG string
        """
        # Color mapping
        colors = {
            "green": "#4c1",      # Clinical success (≥65%)
            "orange": "#fe7d37",  # Above chance but below clinical (<65%)
            "red": "#e05d44",     # At or below chance (≤50%)
            "blue": "#007ec6",    # Info/neutral
            "gray": "#9f9f9f"     # Pending/unknown
        }

        color_hex = colors.get(color, colors["gray"])
        label_width = len(label) * 7 + 10
        value_width = width - label_width

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <clipPath id="a">
        <rect width="{width}" height="20" rx="3" fill="#fff"/>
    </clipPath>
    <g clip-path="url(#a)">
        <path fill="#555" d="M0 0h{label_width}v20H0z"/>
        <path fill="{color_hex}" d="M{label_width} 0h{value_width}v20H{label_width}z"/>
        <path fill="url(#b)" d="M0 0h{width}v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110">
        <text x="{label_width//2 + 1}" y="15" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(label_width-10)*10}">{label}</text>
        <text x="{label_width//2 + 1}" y="14" fill="#fff" transform="scale(.1)" textLength="{(label_width-10)*10}">{label}</text>
        <text x="{label_width + value_width//2 - 1}" y="15" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(value_width-10)*10}">{value}</text>
        <text x="{label_width + value_width//2 - 1}" y="14" fill="#fff" transform="scale(.1)" textLength="{(value_width-10)*10}">{value}</text>
    </g>
</svg>'''
        return svg

    def determine_badge_color(self, balanced_accuracy: float) -> str:
        """Determine badge color based on performance thresholds.

        Args:
            balanced_accuracy: Balanced accuracy value (0.0 to 1.0)

        Returns:
            Color string for badge
        """
        if balanced_accuracy >= self.clinical_threshold:
            return "green"      # Clinical success
        elif balanced_accuracy > self.chance_threshold:
            return "orange"     # Above chance, below clinical
        else:
            return "red"        # At or below chance

    def get_latest_loso_results(self) -> Dict:
        """Load most recent LOSO validation results.

        Returns:
            Dictionary with LOSO results or None if not found
        """
        if not self.results_dir.exists():
            logger.warning("No LOSO results directory found")
            return None

        # Find most recent LOSO results file
        loso_files = list(self.results_dir.glob("loso_validation_*.json"))
        if not loso_files:
            logger.warning("No LOSO validation results found")
            return None

        latest_file = max(loso_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded LOSO results from: {latest_file}")
            return results
        except Exception as e:
            logger.error(f"Failed to load LOSO results: {e}")
            return None

    def extract_performance_metrics(self, results: Dict) -> List[Tuple[str, float, int, int]]:
        """Extract performance metrics from LOSO results.

        Args:
            results: LOSO validation results dictionary

        Returns:
            List of (dataset_name, balanced_accuracy, n_subjects, n_pd) tuples
        """
        metrics = []

        if 'fold_results' in results:
            for fold in results['fold_results']:
                dataset_name = fold['test_dataset']
                ba = fold['metrics']['balanced_accuracy']
                n_subjects = fold['metrics']['n_pd_test'] + fold['metrics']['n_hc_test']
                n_pd = fold['metrics']['n_pd_test']

                metrics.append((dataset_name, ba, n_subjects, n_pd))

        return metrics

    def generate_summary_badges(self, results: Dict) -> None:
        """Generate summary badges for overall LOSO performance.

        Args:
            results: LOSO validation results dictionary
        """
        if 'summary_statistics' not in results:
            logger.warning("No summary statistics found in results")
            return

        summary = results['summary_statistics']
        mean_ba = summary['mean_balanced_accuracy']

        # Overall performance badge
        overall_color = self.determine_badge_color(mean_ba)
        overall_badge = self.create_svg_badge(
            "LOSO Mean BA",
            f"{mean_ba:.1%}",
            overall_color,
            width=140
        )

        overall_file = self.badges_dir / "loso_overall_performance.svg"
        with open(overall_file, 'w') as f:
            f.write(overall_badge)
        logger.info(f"Generated overall performance badge: {overall_file}")

        # Clinical significance badge
        clinical_meets = mean_ba >= self.clinical_threshold
        clinical_color = "green" if clinical_meets else "red"
        clinical_text = "PASS" if clinical_meets else "FAIL"

        clinical_badge = self.create_svg_badge(
            "Clinical (≥65%)",
            clinical_text,
            clinical_color,
            width=130
        )

        clinical_file = self.badges_dir / "loso_clinical_significance.svg"
        with open(clinical_file, 'w') as f:
            f.write(clinical_badge)
        logger.info(f"Generated clinical significance badge: {clinical_file}")

        # Statistical significance badge (if available)
        if 'statistical_test' in summary:
            stat_test = summary['statistical_test']
            is_significant = stat_test.get('significant', False)
            p_value = stat_test.get('p_value', 1.0)

            stat_color = "green" if is_significant else "red"
            stat_text = f"p={p_value:.3f}" if p_value >= 0.001 else "p<0.001"

            stat_badge = self.create_svg_badge(
                "Significance",
                stat_text,
                stat_color,
                width=130
            )

            stat_file = self.badges_dir / "loso_statistical_significance.svg"
            with open(stat_file, 'w') as f:
                f.write(stat_badge)
            logger.info(f"Generated statistical significance badge: {stat_file}")

    def generate_dataset_badges(self, results: Dict) -> None:
        """Generate individual badges for each dataset.

        Args:
            results: LOSO validation results dictionary
        """
        metrics = self.extract_performance_metrics(results)

        for dataset_name, ba, n_subjects, n_pd in metrics:
            # Performance badge for this dataset
            color = self.determine_badge_color(ba)

            performance_badge = self.create_svg_badge(
                f"{dataset_name}",
                f"{ba:.1%}",
                color,
                width=120
            )

            perf_file = self.badges_dir / f"{dataset_name}_performance.svg"
            with open(perf_file, 'w') as f:
                f.write(performance_badge)
            logger.info(f"Generated {dataset_name} performance badge: {perf_file}")

            # Sample size badge
            size_badge = self.create_svg_badge(
                f"{dataset_name}",
                f"{n_subjects}N",
                "blue",
                width=110
            )

            size_file = self.badges_dir / f"{dataset_name}_sample_size.svg"
            with open(size_file, 'w') as f:
                f.write(size_badge)
            logger.info(f"Generated {dataset_name} sample size badge: {size_file}")

    def generate_readme_integration(self, results: Dict) -> str:
        """Generate README integration text with badge links.

        Args:
            results: LOSO validation results dictionary

        Returns:
            Markdown text for README integration
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md_text = f"""
# Phase IV Multi-Site Validation Results

![LOSO Overall Performance](results/badges/loso_overall_performance.svg)
![Clinical Significance](results/badges/loso_clinical_significance.svg)
![Statistical Significance](results/badges/loso_statistical_significance.svg)

## Dataset-Specific Performance

"""

        metrics = self.extract_performance_metrics(results)
        for dataset_name, ba, n_subjects, n_pd in metrics:
            md_text += f"**{dataset_name}**: "
            md_text += f"![Performance](results/badges/{dataset_name}_performance.svg) "
            md_text += f"![Sample Size](results/badges/{dataset_name}_sample_size.svg)\\n"

        md_text += f"""
*Last updated: {timestamp}*

### Badge Legend
- 🟢 **Green**: Clinical threshold met (BA ≥ 65%)
- 🟠 **Orange**: Above chance but below clinical threshold (50% < BA < 65%)
- 🔴 **Red**: At or below chance performance (BA ≤ 50%)
- 🔵 **Blue**: Informational (sample sizes, etc.)

### Automatic Updates
Badges automatically update when new LOSO validation results are generated.
No manual editing required - run validation and badges refresh automatically.
"""

        return md_text

    def run(self) -> None:
        """Main execution function."""
        logger.info("Starting LOSO badge generation...")

        # Load latest results
        results = self.get_latest_loso_results()
        if not results:
            logger.error("No LOSO results found - run validation first")
            return

        # Generate badges
        self.generate_summary_badges(results)
        self.generate_dataset_badges(results)

        # Generate README integration
        readme_text = self.generate_readme_integration(results)
        readme_file = self.badges_dir / "README_integration.md"
        with open(readme_file, 'w') as f:
            f.write(readme_text)
        logger.info(f"Generated README integration: {readme_file}")

        logger.info(f"✅ Badge generation complete! Check {self.badges_dir}/")


def main():
    """Main execution function."""
    generator = LOSOBadgeGenerator()
    generator.run()


if __name__ == "__main__":
    main()