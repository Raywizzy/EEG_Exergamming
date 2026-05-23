#!/usr/bin/env python3
"""
Test script to generate a sample clinical PDF report with robustness analysis
"""

import sys
import json
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.report_pdf import generate_clinical_pdf

def create_sample_robustness_data():
    """Create sample robustness data for testing."""
    return {
        "summary": {
            "total_runs": 64,
            "successful_runs": 62,
            "mean_ba": 89.3,
            "std_ba": 8.2,
            "min_ba": 71.5,
            "max_ba": 97.8,
            "below_threshold": 2,
            "total_runtime_sec": 3600
        },
        "NOISE_LEVEL": {
            "none": {"mean_ba": 97.2, "std_ba": 2.1, "count": 16, "min_ba": 94.1, "max_ba": 99.2},
            "low": {"mean_ba": 94.1, "std_ba": 3.4, "count": 16, "min_ba": 88.7, "max_ba": 98.1},
            "medium": {"mean_ba": 88.7, "std_ba": 5.2, "count": 16, "min_ba": 78.9, "max_ba": 95.4},
            "high": {"mean_ba": 82.4, "std_ba": 7.8, "count": 14, "min_ba": 71.5, "max_ba": 92.1}
        },
        "DROP_CHANNELS": {
            "0": {"mean_ba": 97.2, "std_ba": 2.1, "count": 16, "min_ba": 94.1, "max_ba": 99.2},
            "5": {"mean_ba": 93.8, "std_ba": 3.7, "count": 16, "min_ba": 87.2, "max_ba": 98.4},
            "10": {"mean_ba": 87.6, "std_ba": 6.1, "count": 16, "min_ba": 76.3, "max_ba": 95.1},
            "20": {"mean_ba": 79.1, "std_ba": 9.4, "count": 14, "min_ba": 63.2, "max_ba": 91.7}
        },
        "DURATION_SEC": {
            "0": {"mean_ba": 97.2, "std_ba": 2.1, "count": 21, "min_ba": 94.1, "max_ba": 99.2},
            "120": {"mean_ba": 94.7, "std_ba": 3.9, "count": 21, "min_ba": 86.2, "max_ba": 98.8},
            "60": {"mean_ba": 88.3, "std_ba": 7.2, "count": 20, "min_ba": 71.5, "max_ba": 96.4}
        },
        "DOWNSAMPLE_RATE": {
            "256": {"mean_ba": 97.2, "std_ba": 2.1, "count": 21, "min_ba": 94.1, "max_ba": 99.2},
            "128": {"mean_ba": 92.5, "std_ba": 4.3, "count": 21, "min_ba": 83.7, "max_ba": 97.9},
            "64": {"mean_ba": 84.1, "std_ba": 8.7, "count": 20, "min_ba": 68.4, "max_ba": 94.3}
        }
    }

def main():
    """Generate sample report with robustness data."""
    print("=" * 60)
    print("📄 Testing Clinical PDF Report with Robustness Analysis")
    print("=" * 60)

    # Create sample job data
    job_info = {
        'job_id': 'test-report-12345678',
        'ba': 93.5,
        'p_value': 0.001,
        'effect_size': 8.7,
        'runtime_sec': 0.38,
        'status': 'succeeded'
    }

    job_data = {
        'processing_stats': {
            'subjects_processed': 15,
            'total_epochs': 4500,
            'artifacts_removed': 235
        }
    }

    # Create sample robustness data file
    robustness_data = create_sample_robustness_data()
    robustness_dir = Path("data/out/robustness")
    robustness_dir.mkdir(parents=True, exist_ok=True)

    # Save sample robustness data
    with (robustness_dir / "robustness_panel.json").open('w') as f:
        json.dump(robustness_data, f, indent=2)

    print("✅ Created sample robustness data")

    # Generate PDF report
    output_path = Path("sample_robustness_report.pdf")

    try:
        generate_clinical_pdf(
            output_path=str(output_path),
            site_id="TEST_SITE",
            trial_id="ROBUSTNESS_DEMO",
            job_data=job_data,
            job_info=job_info
        )

        print(f"✅ Generated sample report: {output_path}")

        # Also create HTML version for easy viewing
        html_path = str(output_path).replace('.pdf', '.html')
        if Path(html_path).exists():
            print(f"📄 HTML version available: {html_path}")

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False

    print("\n" + "=" * 60)
    print("📊 ROBUSTNESS REPORT FEATURES")
    print("=" * 60)

    print("\n🧪 Robustness Analysis Section Includes:")
    print("• Traffic light indicators for each factor (🟢🟡🟠🔴)")
    print("• Comprehensive performance statistics")
    print("• Factor-specific performance tables")
    print("• Embedded plots (when available)")
    print("• Clinical interpretation with deployment confidence")

    print("\n🎯 Traffic Light System:")
    print("• 🟢 Excellent: <5% performance degradation")
    print("• 🟡 Good: 5-15% performance degradation")
    print("• 🟠 Moderate: 15-25% performance degradation")
    print("• 🔴 Limited: >25% performance degradation")

    print("\n📈 Key Metrics Displayed:")
    print("• Total robustness tests run")
    print("• Overall performance range (min-max)")
    print("• Tests below clinical threshold (65%)")
    print("• Deployment confidence assessment")

    print("\n🏥 Clinical Value:")
    print("• Evidence-based deployment decisions")
    print("• Quantified confidence metrics")
    print("• Regulatory compliance documentation")
    print("• Risk assessment for clinical use")

    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)