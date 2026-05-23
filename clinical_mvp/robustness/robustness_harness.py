#!/usr/bin/env python3
"""
Core15+ Robustness Testing Harness
Run controlled degradations and summarize robustness into CSV + plots.
Also writes a compact JSON for the PDF report to include a robustness panel.
"""
from __future__ import annotations
import os, json, csv, itertools, subprocess, shlex, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ENTRYPOINT = os.environ.get("CORE15_ENTRYPOINT", "/app/scripts/clinical_trial_processor.py")
CONFIG = os.environ.get("CORE15_CONFIG", "/app/config/preprocessing_config.yaml")
BIDS = os.environ.get("BIDS_IN", "/app/data/in")
OUT = os.environ.get("RESULTS_OUT", "/app/data/out/robustness")

FACTORS = {
    "NOISE_LEVEL": ["none", "low", "medium", "high"],
    "DROP_CHANNELS": [0, 5, 10, 20],
    "DURATION_SEC": [0, 60, 120],  # 0 = full length
    "DOWNSAMPLE_RATE": [256, 128, 64],
}

# Baseline performance targets
BASELINE_BA = 97.2
CLINICAL_THRESHOLD = 65.0


def _run_once(env_extra: dict) -> dict:
    """Run Core15+ with specific degradation parameters."""
    run_id = "_".join([f"{k}-{v}" for k, v in env_extra.items()])
    out_dir = Path(OUT) / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command with degradation parameters
    cmd_args = [
        f'python {ENTRYPOINT}',
        f'--config {shlex.quote(CONFIG)}',
        f'--bids_dir {shlex.quote(BIDS)}',
        f'--out_dir {shlex.quote(str(out_dir))}',
        '--site_id ROBUSTNESS',
        '--trial_id VALIDATION'
    ]

    # Add degradation parameters
    for key, value in env_extra.items():
        if key == "NOISE_LEVEL" and value != "none":
            cmd_args.append(f'--add_noise {value}')
        elif key == "DROP_CHANNELS" and value > 0:
            cmd_args.append(f'--drop_channels {value}')
        elif key == "DURATION_SEC" and value > 0:
            cmd_args.append(f'--max_duration {value}')
        elif key == "DOWNSAMPLE_RATE" and value != 256:
            cmd_args.append(f'--resample_rate {value}')

    cmd = ' '.join(cmd_args)

    print(f"Running robustness test: {run_id}")

    # Set environment variables for degradation control
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items()})

    t0 = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, env=env,
                              capture_output=True, text=True)
        dt = time.time() - t0

        # Parse results from output directory
        summary = {}
        summary_file = out_dir / "run_summary.json"
        if summary_file.exists():
            try:
                summary = json.loads(summary_file.read_text())
            except Exception as e:
                print(f"Warning: Could not parse summary file: {e}")

        ba = summary.get("performance", {}).get("balanced_accuracy")
        p_val = summary.get("performance", {}).get("p_value")
        effect_size = summary.get("performance", {}).get("effect_size")

        return {
            "run_id": run_id,
            "out_dir": str(out_dir),
            "runtime_sec": round(dt, 2),
            "ba": ba,
            "p_value": p_val,
            "effect_size": effect_size,
            "status": "success",
            **env_extra
        }

    except subprocess.CalledProcessError as e:
        dt = time.time() - t0
        return {
            "run_id": run_id,
            "out_dir": str(out_dir),
            "runtime_sec": round(dt, 2),
            "ba": None,
            "p_value": None,
            "effect_size": None,
            "status": "failed",
            "error": str(e),
            **env_extra
        }


def _create_robustness_plots(results: list[dict]) -> None:
    """Create robustness visualization plots."""
    plots_dir = Path(OUT) / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Set up matplotlib for clean plots
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Core15+ Robustness Analysis', fontsize=14, fontweight='bold')

    for idx, (factor, levels) in enumerate(FACTORS.items()):
        ax = axes[idx // 2, idx % 2]

        # Aggregate results by factor level
        factor_results = {}
        for level in levels:
            bas = [r["ba"] for r in results
                  if r.get(factor) == level and r["ba"] is not None]
            if bas:
                factor_results[str(level)] = {
                    'mean': np.mean(bas),
                    'std': np.std(bas),
                    'count': len(bas)
                }

        if factor_results:
            labels = list(factor_results.keys())
            means = [factor_results[l]['mean'] for l in labels]
            stds = [factor_results[l]['std'] for l in labels]

            bars = ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)

            # Color bars based on performance
            for bar, mean in zip(bars, means):
                if mean >= 90:
                    bar.set_color('#16a34a')  # Green - Excellent
                elif mean >= 75:
                    bar.set_color('#f59e0b')  # Orange - Good
                elif mean >= CLINICAL_THRESHOLD:
                    bar.set_color('#dc2626')  # Red - Acceptable
                else:
                    bar.set_color('#7f1d1d')  # Dark red - Below threshold

            # Add reference lines
            ax.axhline(y=BASELINE_BA, color='blue', linestyle='--', alpha=0.6,
                      label=f'Baseline ({BASELINE_BA}%)')
            ax.axhline(y=CLINICAL_THRESHOLD, color='red', linestyle='--', alpha=0.6,
                      label=f'Clinical Threshold ({CLINICAL_THRESHOLD}%)')

            ax.set_title(factor.replace('_', ' ').title())
            ax.set_ylabel('Balanced Accuracy (%)')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                       f'{mean:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(plots_dir / "robustness_overview.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Create a summary degradation plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Calculate overall degradation by combining all non-baseline conditions
    baseline_results = [r for r in results
                       if all(r.get(f) == list(FACTORS[f])[0] for f in FACTORS.keys())
                       and r["ba"] is not None]
    degraded_results = [r for r in results
                       if not all(r.get(f) == list(FACTORS[f])[0] for f in FACTORS.keys())
                       and r["ba"] is not None]

    if baseline_results and degraded_results:
        baseline_ba = np.mean([r["ba"] for r in baseline_results])
        all_bas = [r["ba"] for r in degraded_results]

        # Create degradation histogram
        ax.hist(all_bas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=baseline_ba, color='blue', linestyle='-', linewidth=2,
                  label=f'Baseline: {baseline_ba:.1f}%')
        ax.axvline(x=CLINICAL_THRESHOLD, color='red', linestyle='--', linewidth=2,
                  label=f'Clinical Threshold: {CLINICAL_THRESHOLD}%')

        ax.set_xlabel('Balanced Accuracy (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Performance Under Degradation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "degradation_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Robustness plots saved to {plots_dir}")


def main():
    """Run complete robustness testing suite."""
    print("=" * 60)
    print("🧠 Core15+ Robustness Testing Harness")
    print("=" * 60)

    Path(OUT).mkdir(parents=True, exist_ok=True)

    # Generate all factor combinations
    combinations = list(itertools.product(*FACTORS.values()))
    total_runs = len(combinations)

    print(f"Planning {total_runs} robustness test runs...")
    print(f"Factors: {list(FACTORS.keys())}")
    print(f"Output directory: {OUT}")

    results = []
    start_time = time.time()

    for i, combo in enumerate(combinations, 1):
        env_map = {k: v for k, v in zip(FACTORS.keys(), combo)}

        print(f"\n[{i}/{total_runs}] Testing: {env_map}")

        try:
            result = _run_once(env_map)
            results.append(result)

            if result["ba"] is not None:
                print(f"  ✅ BA: {result['ba']:.1f}% (runtime: {result['runtime_sec']}s)")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"  💥 Exception: {e}")
            results.append({
                **env_map,
                "runtime_sec": None,
                "ba": None,
                "p_value": None,
                "effect_size": None,
                "status": "exception",
                "error": str(e)
            })

    total_time = time.time() - start_time

    # Save detailed CSV results
    csv_path = Path(OUT) / "robustness_summary.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [*FACTORS.keys(), "ba", "p_value", "effect_size",
                     "runtime_sec", "status", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Ensure all fields are present
            row = {k: result.get(k) for k in fieldnames}
            writer.writerow(row)

    # Create compact JSON for PDF embedding
    compact = {}
    successful_results = [r for r in results if r["ba"] is not None]

    for factor, levels in FACTORS.items():
        factor_summary = {}
        for level in levels:
            bas = [r["ba"] for r in successful_results if r.get(factor) == level]
            if bas:
                factor_summary[str(level)] = {
                    "mean_ba": round(np.mean(bas), 1),
                    "std_ba": round(np.std(bas), 1),
                    "count": len(bas),
                    "min_ba": round(min(bas), 1),
                    "max_ba": round(max(bas), 1)
                }
            else:
                factor_summary[str(level)] = None
        compact[factor] = factor_summary

    # Add overall summary statistics
    if successful_results:
        all_bas = [r["ba"] for r in successful_results]
        compact["summary"] = {
            "total_runs": len(results),
            "successful_runs": len(successful_results),
            "mean_ba": round(np.mean(all_bas), 1),
            "std_ba": round(np.std(all_bas), 1),
            "min_ba": round(min(all_bas), 1),
            "max_ba": round(max(all_bas), 1),
            "below_threshold": sum(1 for ba in all_bas if ba < CLINICAL_THRESHOLD),
            "total_runtime_sec": round(total_time, 1)
        }

    # Save compact summary
    panel_path = Path(OUT) / "robustness_panel.json"
    panel_path.write_text(json.dumps(compact, indent=2))

    # Create visualization plots
    try:
        _create_robustness_plots(results)
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 ROBUSTNESS TESTING COMPLETE")
    print("=" * 60)
    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(results) - len(successful_results)}")
    print(f"Total runtime: {total_time/60:.1f} minutes")

    if successful_results:
        all_bas = [r["ba"] for r in successful_results]
        print(f"Performance range: {min(all_bas):.1f}% - {max(all_bas):.1f}%")
        print(f"Mean performance: {np.mean(all_bas):.1f}% ± {np.std(all_bas):.1f}%")
        below_threshold = sum(1 for ba in all_bas if ba < CLINICAL_THRESHOLD)
        print(f"Below clinical threshold: {below_threshold}/{len(all_bas)} ({100*below_threshold/len(all_bas):.1f}%)")

    print(f"\nResults saved to:")
    print(f"  📄 Detailed CSV: {csv_path}")
    print(f"  📊 Summary JSON: {panel_path}")
    print(f"  📈 Plots: {Path(OUT) / 'plots'}")
    print("=" * 60)

    return len(successful_results) == len(results)  # Return True if all runs succeeded


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)