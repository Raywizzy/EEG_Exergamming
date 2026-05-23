#!/usr/bin/env python3
"""
Manuscript Badge Synchronization System
Automatically updates manuscript templates with real-time validation status badges
"""

import os
import re
import glob
from pathlib import Path
from datetime import datetime
import yaml

class ManuscriptBadgeSync:
    def __init__(self, project_root="/Users/user/Desktop/EEG_Exergamming"):
        self.project_root = Path(project_root)
        self.badge_dir = self.project_root / "results" / "badges"
        self.manuscript_dir = self.project_root / "manuscripts"

        # Badge status mappings
        self.badge_status = {
            "loso_overall_performance.svg": "Overall Performance",
            "loso_clinical_significance.svg": "Clinical Significance",
            "ds004584_performance.svg": "Iowa Performance",
            "ds002778_performance.svg": "UCSD Performance",
            "coral_adaptation.svg": "CORAL Status",
            "multisite_validation.svg": "Multi-Site Validation"
        }

    def check_badge_status(self):
        """Check current status of all validation badges"""
        badge_status = {}

        for badge_file, description in self.badge_status.items():
            badge_path = self.badge_dir / badge_file
            if badge_path.exists():
                # Read SVG to determine status color
                with open(badge_path, 'r') as f:
                    svg_content = f.read()

                # Extract color to determine status
                if '#4c1' in svg_content:  # Green
                    status = "✅ PASS"
                elif '#fe7d37' in svg_content:  # Orange
                    status = "⚠️ CONDITIONAL"
                elif '#e05d44' in svg_content:  # Red
                    status = "❌ FAIL"
                else:
                    status = "⚪ UNKNOWN"

                badge_status[description] = {
                    "file": badge_file,
                    "status": status,
                    "path": f"../results/badges/{badge_file}",
                    "exists": True
                }
            else:
                badge_status[description] = {
                    "file": badge_file,
                    "status": "⚪ MISSING",
                    "path": f"../results/badges/{badge_file}",
                    "exists": False
                }

        return badge_status

    def update_infrastructure_paper(self):
        """Update Infrastructure Framework Paper with current badge status"""
        paper_path = self.manuscript_dir / "Infrastructure_Framework_Paper.md"

        if not paper_path.exists():
            print(f"❌ Infrastructure paper not found at {paper_path}")
            return False

        with open(paper_path, 'r') as f:
            content = f.read()

        # Update timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = re.sub(
            r'\*\*Last Updated\*\*:.*',
            f'**Last Updated**: {timestamp}',
            content
        )

        # Update badge references to ensure they're pointing to correct paths
        badge_updates = [
            (r'!\[Infrastructure Status\]\([^)]+\)', '![Infrastructure Status](../results/badges/multisite_validation.svg)'),
            (r'!\[Badge System\]\([^)]+\)', '![Badge System](../results/badges/coral_adaptation.svg)'),
            (r'!\[Overall Performance\]\([^)]+\)', '![Overall Performance](../results/badges/loso_overall_performance.svg)'),
            (r'!\[Clinical Significance\]\([^)]+\)', '![Clinical Significance](../results/badges/loso_clinical_significance.svg)')
        ]

        for pattern, replacement in badge_updates:
            content = re.sub(pattern, replacement, content)

        # Write updated content
        with open(paper_path, 'w') as f:
            f.write(content)

        print(f"✅ Updated Infrastructure Framework Paper - {timestamp}")
        return True

    def update_biomarker_paper(self):
        """Update Biomarker Validation Paper with current badge status"""
        paper_path = self.manuscript_dir / "Biomarker_Validation_Paper.md"

        if not paper_path.exists():
            print(f"❌ Biomarker paper not found at {paper_path}")
            return False

        with open(paper_path, 'r') as f:
            content = f.read()

        # Update timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = re.sub(
            r'\*\*Last Updated\*\*:.*',
            f'**Last Updated**: {timestamp}',
            content
        )

        # Update badge references
        badge_updates = [
            (r'!\[LOSO Overall Performance\]\([^)]+\)', '![LOSO Overall Performance](../results/badges/loso_overall_performance.svg)'),
            (r'!\[Clinical Significance\]\([^)]+\)', '![Clinical Significance](../results/badges/loso_clinical_significance.svg)'),
            (r'!\[Iowa Performance\]\([^)]+\)', '![Iowa Performance](../results/badges/ds004584_performance.svg)'),
            (r'!\[UCSD Performance\]\([^)]+\)', '![UCSD Performance](../results/badges/ds002778_performance.svg)'),
            (r'!\[CORAL Adaptation\]\([^)]+\)', '![CORAL Adaptation](../results/badges/coral_adaptation.svg)'),
            (r'!\[Multi-Site Validation\]\([^)]+\)', '![Multi-Site Validation](../results/badges/multisite_validation.svg)')
        ]

        for pattern, replacement in badge_updates:
            content = re.sub(pattern, replacement, content)

        # Write updated content
        with open(paper_path, 'w') as f:
            f.write(content)

        print(f"✅ Updated Biomarker Validation Paper - {timestamp}")
        return True

    def update_main_manuscript(self):
        """Update the main Phase IV manuscript"""
        paper_path = self.manuscript_dir / "PhaseIV_MultiSite_Validation.md"

        if not paper_path.exists():
            print(f"❌ Main manuscript not found at {paper_path}")
            return False

        with open(paper_path, 'r') as f:
            content = f.read()

        # Update timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = re.sub(
            r'\*\*Last Updated\*\*:.*',
            f'**Last Updated**: {timestamp}',
            content
        )

        # Write updated content
        with open(paper_path, 'w') as f:
            f.write(content)

        print(f"✅ Updated Main Phase IV Manuscript - {timestamp}")
        return True

    def generate_status_report(self):
        """Generate comprehensive status report for all manuscripts"""
        badge_status = self.check_badge_status()

        print("\n" + "="*80)
        print("📊 MANUSCRIPT BADGE SYNCHRONIZATION REPORT")
        print("="*80)
        print(f"🕒 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Project: {self.project_root}")

        print("\n📋 BADGE STATUS SUMMARY:")
        print("-" * 50)
        for description, info in badge_status.items():
            print(f"{info['status']} {description:<25} → {info['file']}")

        print("\n📄 MANUSCRIPT UPDATE STATUS:")
        print("-" * 50)

        # Check manuscript files
        manuscripts = [
            ("Infrastructure Framework", "Infrastructure_Framework_Paper.md"),
            ("Biomarker Validation", "Biomarker_Validation_Paper.md"),
            ("Phase IV Main", "PhaseIV_MultiSite_Validation.md")
        ]

        for name, filename in manuscripts:
            path = self.manuscript_dir / filename
            if path.exists():
                # Get last modified time
                mod_time = datetime.fromtimestamp(path.stat().st_mtime)
                print(f"✅ {name:<20} → Updated {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"❌ {name:<20} → FILE MISSING")

        print("\n🔄 INTEGRATION STATUS:")
        print("-" * 50)

        # Count valid badge links
        total_badges = len(self.badge_status)
        existing_badges = sum(1 for info in badge_status.values() if info['exists'])
        print(f"📊 Badge Coverage: {existing_badges}/{total_badges} ({existing_badges/total_badges*100:.1f}%)")

        # Check if manuscripts have proper badge integration
        for name, filename in manuscripts:
            path = self.manuscript_dir / filename
            if path.exists():
                with open(path, 'r') as f:
                    content = f.read()
                badge_count = len(re.findall(r'!\[.*?\]\(\.\.\/results\/badges\/.*?\.svg\)', content))
                print(f"📋 {name}: {badge_count} badge references")

        print("\n" + "="*80)

        return badge_status

    def sync_all_manuscripts(self):
        """Synchronize all manuscripts with current badge status"""
        print("🚀 Starting manuscript badge synchronization...")

        # Update all manuscripts
        results = []
        results.append(self.update_infrastructure_paper())
        results.append(self.update_biomarker_paper())
        results.append(self.update_main_manuscript())

        # Generate status report
        self.generate_status_report()

        success_count = sum(results)
        total_count = len(results)

        print(f"\n🎯 SYNCHRONIZATION COMPLETE: {success_count}/{total_count} manuscripts updated")

        if success_count == total_count:
            print("✅ All manuscripts successfully synchronized with badge system")
        else:
            print("⚠️ Some manuscripts failed to update - check file paths and permissions")

        return success_count == total_count

if __name__ == "__main__":
    # Initialize badge sync system
    sync_system = ManuscriptBadgeSync()

    # Run full synchronization
    success = sync_system.sync_all_manuscripts()

    # Exit with appropriate code
    exit(0 if success else 1)