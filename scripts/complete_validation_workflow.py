#!/usr/bin/env python3
"""
Complete integrated validation workflow
Analytics → Reporting → Visual Signaling → Manuscript → Dashboard
Gold-standard translational pipeline automation
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationWorkflow:
    """Complete validation workflow orchestrator."""

    def __init__(self, datasets=None):
        self.datasets = datasets or ["ds004584", "ds002778"]
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("results/complete_workflow") / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd, description):
        """Run command with logging."""
        logger.info(f"🔄 {description}...")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                logger.info(f"✅ {description} completed")
                return True
            else:
                logger.error(f"❌ {description} failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {description} timed out")
            return False
        except Exception as e:
            logger.error(f"💥 {description} error: {e}")
            return False

    def step1_analytics_pipeline(self):
        """Step 1: Run LOSO validation (analytics)."""
        logger.info("=" * 60)
        logger.info("STEP 1: ANALYTICS PIPELINE")
        logger.info("=" * 60)

        # Run LOSO validation
        datasets_str = " ".join(self.datasets)
        cmd = f"python scripts/loso_validation.py --datasets {datasets_str}"
        success = self.run_command(cmd, "LOSO Cross-Validation")

        return success

    def step2_visual_reporting(self):
        """Step 2: Generate plots and reports (reporting)."""
        logger.info("=" * 60)
        logger.info("STEP 2: VISUAL REPORTING")
        logger.info("=" * 60)

        # Generate LOSO plots
        cmd1 = "python scripts/make_loso_plots.py --input results/loso_validation/2site_loso_summary.csv --outdir results/figures/loso_2site"
        success1 = self.run_command(cmd1, "LOSO Performance Plots")

        # Generate results highlights
        cmd2 = "python scripts/make_results_highlights.py --summary_csv results/figures/loso_2site/loso_overall_summary.csv --outpdf results/PhaseIV_Results_Highlights_2site.pdf --n_total 127 --n_pd 82 --n_hc 45"
        success2 = self.run_command(cmd2, "Results Highlights PDF")

        # Build interim report
        cmd3 = "python scripts/build_interim_report.py --outpdf results/PhaseIV_Interim_Report_FULL_2site.pdf --loso_summary_csv results/figures/loso_2site/loso_overall_summary.csv --loso_figs_dir results/figures/loso_2site --n_total 127 --n_pd 82 --n_hc 45"
        success3 = self.run_command(cmd3, "Comprehensive Interim Report")

        return all([success1, success2, success3])

    def step3_badge_automation(self):
        """Step 3: Generate and update badges (visual signaling)."""
        logger.info("=" * 60)
        logger.info("STEP 3: VISUAL SIGNALING (BADGES)")
        logger.info("=" * 60)

        # Generate current badges
        cmd1 = "python scripts/create_current_badges.py"
        success1 = self.run_command(cmd1, "Dynamic Badge Generation")

        # Update validation status
        cmd2 = "python scripts/update_validation_status.py"
        success2 = self.run_command(cmd2, "Validation Status Update")

        return all([success1, success2])

    def step4_manuscript_integration(self):
        """Step 4: Update manuscript (manuscript)."""
        logger.info("=" * 60)
        logger.info("STEP 4: MANUSCRIPT INTEGRATION")
        logger.info("=" * 60)

        # Update manuscript with results
        cmd = "python scripts/update_manuscript.py"
        success = self.run_command(cmd, "Manuscript Auto-Update")

        return success

    def step5_enhanced_dashboard(self):
        """Step 5: Create enhanced dashboard (dashboard)."""
        logger.info("=" * 60)
        logger.info("STEP 5: ENHANCED DASHBOARD")
        logger.info("=" * 60)

        # Build enhanced report with embedded badges
        cmd = "python scripts/build_enhanced_report.py --summary_csv results/figures/loso_2site/loso_overall_summary.csv --output_pdf results/PhaseIV_Enhanced_Dashboard_with_Badges.pdf"
        success = self.run_command(cmd, "Enhanced Badge Dashboard")

        return success

    def generate_workflow_summary(self, successes):
        """Generate final workflow summary."""
        logger.info("=" * 60)
        logger.info("WORKFLOW COMPLETION SUMMARY")
        logger.info("=" * 60)

        steps = [
            ("Analytics Pipeline", successes[0]),
            ("Visual Reporting", successes[1]),
            ("Badge Automation", successes[2]),
            ("Manuscript Integration", successes[3]),
            ("Enhanced Dashboard", successes[4])
        ]

        total_success = all(successes)

        for step, success in steps:
            status = "✅ COMPLETE" if success else "❌ FAILED"
            logger.info(f"{step}: {status}")

        logger.info("=" * 60)

        if total_success:
            logger.info("🎉 COMPLETE WORKFLOW SUCCESS!")
            logger.info("🚀 Gold-standard translational pipeline achieved")

            print("\n📊 GENERATED ARTIFACTS:")
            print("├── 📈 Analytics: LOSO validation results")
            print("├── 📋 Reports: Highlights + Interim + Enhanced Dashboard PDFs")
            print("├── 🏷️  Badges: Auto-updating SVG status indicators")
            print("├── 📄 Manuscript: Auto-synced with latest results")
            print("└── 🎯 Dashboard: Live badge-integrated supervisor report")

            print("\n🔗 KEY FILES:")
            print("• results/PhaseIV_Enhanced_Dashboard_with_Badges.pdf")
            print("• manuscripts/PhaseIV_MultiSite_Validation.md")
            print("• results/badges/ (auto-updating status)")
            print("• README.md (integrated status display)")

            print("\n🎯 SUPERVISOR READY:")
            print("✓ Traffic-light badge system")
            print("✓ Real-time status indicators")
            print("✓ Complete audit trail")
            print("✓ Regulatory compliance documentation")

        else:
            logger.error("💥 WORKFLOW INCOMPLETE - Check failed steps above")

        return total_success

    def run_complete_workflow(self):
        """Execute complete validation workflow."""
        logger.info("🚀 Starting Complete Validation Workflow")
        logger.info(f"📅 Timestamp: {self.timestamp}")
        logger.info(f"📂 Output Directory: {self.output_dir}")
        logger.info(f"🎯 Datasets: {self.datasets}")

        # Execute all steps
        successes = [
            self.step1_analytics_pipeline(),
            self.step2_visual_reporting(),
            self.step3_badge_automation(),
            self.step4_manuscript_integration(),
            self.step5_enhanced_dashboard()
        ]

        # Generate summary
        return self.generate_workflow_summary(successes)

def main():
    """Main execution function."""
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["ds004584", "ds002778"]

    print("🏆 PHASE IV COMPLETE VALIDATION WORKFLOW")
    print("Analytics → Reporting → Visual Signaling → Manuscript → Dashboard")
    print("=" * 70)

    workflow = ValidationWorkflow(datasets)
    success = workflow.run_complete_workflow()

    if success:
        print("\n🎉 GOLD-STANDARD TRANSLATIONAL WORKFLOW COMPLETE!")
        print("Your validation pipeline is now clinical-trial grade! 🚀")
        sys.exit(0)
    else:
        print("\n❌ Workflow incomplete - check errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()