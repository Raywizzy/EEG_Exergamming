#!/usr/bin/env python3
"""
Create badges from current LOSO validation results
ds004584: 60.3% BA, ds002778: 70.8% BA
"""

import sys
sys.path.append('.')
from scripts.generate_loso_badges import LOSOBadgeGenerator
from pathlib import Path

# Create badges directory
badges_dir = Path("results/badges")
badges_dir.mkdir(parents=True, exist_ok=True)

generator = LOSOBadgeGenerator()

# Current results from our successful LOSO validation
ds004584_ba = 0.603  # 60.3%
ds002778_ba = 0.708  # 70.8%
mean_ba = (ds004584_ba + ds002778_ba) / 2  # 65.55%

print("Creating badges for current LOSO validation results...")
print(f"ds004584: {ds004584_ba:.1%} BA")
print(f"ds002778: {ds002778_ba:.1%} BA")
print(f"Mean: {mean_ba:.1%} BA")

# Overall performance badge
overall_color = generator.determine_badge_color(mean_ba)
overall_badge = generator.create_svg_badge(
    "LOSO Mean BA",
    f"{mean_ba:.1%}",
    overall_color,
    width=140
)

overall_file = badges_dir / "loso_overall_performance.svg"
with open(overall_file, 'w') as f:
    f.write(overall_badge)
print(f"✅ Generated: {overall_file}")

# Clinical significance badge
clinical_meets = mean_ba >= 0.65
clinical_color = "green" if clinical_meets else "red"
clinical_text = "PASS" if clinical_meets else "FAIL"

clinical_badge = generator.create_svg_badge(
    "Clinical (≥65%)",
    clinical_text,
    clinical_color,
    width=130
)

clinical_file = badges_dir / "loso_clinical_significance.svg"
with open(clinical_file, 'w') as f:
    f.write(clinical_badge)
print(f"✅ Generated: {clinical_file}")

# Individual dataset badges
datasets = [
    ("ds004584", ds004584_ba, 117),  # (name, BA, sample_size)
    ("ds002778", ds002778_ba, 10)
]

for dataset_name, ba, n_subjects in datasets:
    # Performance badge
    color = generator.determine_badge_color(ba)
    performance_badge = generator.create_svg_badge(
        f"{dataset_name}",
        f"{ba:.1%}",
        color,
        width=120
    )

    perf_file = badges_dir / f"{dataset_name}_performance.svg"
    with open(perf_file, 'w') as f:
        f.write(performance_badge)
    print(f"✅ Generated: {perf_file}")

    # Sample size badge
    size_badge = generator.create_svg_badge(
        f"{dataset_name}",
        f"{n_subjects}N",
        "blue",
        width=110
    )

    size_file = badges_dir / f"{dataset_name}_sample_size.svg"
    with open(size_file, 'w') as f:
        f.write(size_badge)
    print(f"✅ Generated: {size_file}")

# Multi-site validation badge
multisite_badge = generator.create_svg_badge(
    "Multi-Site",
    "2 Sites",
    "green",
    width=110
)

multisite_file = badges_dir / "multisite_validation.svg"
with open(multisite_file, 'w') as f:
    f.write(multisite_badge)
print(f"✅ Generated: {multisite_file}")

# CORAL domain adaptation badge
coral_badge = generator.create_svg_badge(
    "CORAL",
    "Active",
    "green",
    width=100
)

coral_file = badges_dir / "coral_adaptation.svg"
with open(coral_file, 'w') as f:
    f.write(coral_badge)
print(f"✅ Generated: {coral_file}")

print(f"\n🎉 All badges generated in {badges_dir}/")
print("\nBadge meanings:")
print("🟢 Green: Clinical threshold met (BA ≥ 65%)")
print("🟠 Orange: Above chance but below clinical (50% < BA < 65%)")
print("🔴 Red: At or below chance (BA ≤ 50%)")
print("🔵 Blue: Informational")