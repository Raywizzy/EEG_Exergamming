#!/usr/bin/env python3
"""
Update badges for 3-site LOSO validation results
Clinical-trial grade badge generation
"""

import json
from pathlib import Path

def create_3site_badges():
    """Create badges for 3-site LOSO validation"""

    # Load 3-site results
    stats_file = Path('/Users/user/Desktop/EEG_Exergamming/results/statistical_analysis/latest_comprehensive_statistics.json')

    if not stats_file.exists():
        print("❌ 3-site statistics not found")
        return

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    mean_ba = stats['mean_ba']
    p_value = stats['p_value']
    n_sites = stats['n_folds']
    n_subjects = stats['n_subjects_total']

    # Create badges
    badges_dir = Path('/Users/user/Desktop/EEG_Exergamming/results/badges')
    badges_dir.mkdir(parents=True, exist_ok=True)

    # 3-site validation badge
    ba_percent = mean_ba * 100
    color = "green" if mean_ba >= 0.65 else "orange" if mean_ba >= 0.5 else "red"

    badge_3site = f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="20">
  <defs>
    <linearGradient id="b" x2="0" y2="100%">
      <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
      <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
  </defs>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <rect width="200" height="20" fill="#555"/>
    <rect x="90" width="110" height="20" fill="{color}"/>
    <rect width="200" height="20" fill="url(#b)"/>
    <text x="45" y="15" fill="#fff">3-Site LOSO</text>
    <text x="145" y="15" fill="#fff">{ba_percent:.1f}% BA</text>
  </g>
</svg>"""

    with open(badges_dir / '3site_loso_validation.svg', 'w') as f:
        f.write(badge_3site)

    # Statistical significance badge
    sig_color = "green" if p_value < 0.05 else "orange" if p_value < 0.1 else "red"
    sig_text = "SIGNIFICANT" if p_value < 0.05 else f"p={p_value:.3f}"

    badge_sig = f"""<svg xmlns="http://www.w3.org/2000/svg" width="220" height="20">
  <defs>
    <linearGradient id="b" x2="0" y2="100%">
      <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
      <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
  </defs>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <rect width="220" height="20" fill="#555"/>
    <rect x="90" width="130" height="20" fill="{sig_color}"/>
    <rect width="220" height="20" fill="url(#b)"/>
    <text x="45" y="15" fill="#fff">Statistical</text>
    <text x="155" y="15" fill="#fff">{sig_text}</text>
  </g>
</svg>"""

    with open(badges_dir / 'statistical_significance.svg', 'w') as f:
        f.write(badge_sig)

    # Cross-site validation badge
    badge_cross = f"""<svg xmlns="http://www.w3.org/2000/svg" width="180" height="20">
  <defs>
    <linearGradient id="b" x2="0" y2="100%">
      <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
      <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
  </defs>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <rect width="180" height="20" fill="#555"/>
    <rect x="80" width="100" height="20" fill="blue"/>
    <rect width="180" height="20" fill="url(#b)"/>
    <text x="40" y="15" fill="#fff">Cross-site</text>
    <text x="130" y="15" fill="#fff">{n_sites} sites, {n_subjects} subjects</text>
  </g>
</svg>"""

    with open(badges_dir / 'cross_site_validation.svg', 'w') as f:
        f.write(badge_cross)

    print("✅ Generated 3-site validation badges:")
    print(f"  • 3-site LOSO: {ba_percent:.1f}% BA ({color})")
    print(f"  • Statistical: {sig_text} ({sig_color})")
    print(f"  • Cross-site: {n_sites} sites, {n_subjects} subjects")

if __name__ == "__main__":
    create_3site_badges()