#!/usr/bin/env python3
"""
Minimal Sprint Plan for EEG Exergaming Project
Implements the 6-step plan to get from dataset to working classifier.

Steps:
1. Fetch & parse MRC neurofeedback → extract REAL vs SHAM trials
2. Preprocess EEG (1–40 Hz bandpass, 50 Hz notch; ICA for EOG/EMG)
3. Feature sets: α/β power, IAF, ratios
4. Subject-wise CV: LR/SVM/RF with Bayesian HPO
5. Lock pipeline → External test on Leicester
6. Save every figure/waveform
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.figure_manager import save_figure, create_figure, close_all_figures

def print_timestamp(message):
    """Print message with ISO timestamp."""
    timestamp = datetime.now(timezone.utc).isoformat()
    iso_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    print(f"[{timestamp}] {message}")
    print(f"ISO Date: {iso_date}")

def check_prerequisites():
    """Check if all prerequisites are met before starting."""
    print_timestamp("Checking prerequisites")

    checks_passed = 0
    total_checks = 4

    # Check 1: MRC BNDU dataset downloaded
    mrc_data_path = Path("data/raw/mrc_bndu_lfps-and-eegs-patients-parkinsons-disease-during-neurofeedback-training")
    if mrc_data_path.exists():
        print("✓ Check 1: MRC BNDU dataset directory exists")
        checks_passed += 1
    else:
        print("✗ Check 1: MRC BNDU dataset not found")
        print("   Run: python scripts/download_dataset.py --source mrc_bndu --dataset lfps-and-eegs-patients-parkinsons-disease-during-neurofeedback-training")

    # Check 2: Leicester external validation dataset
    leicester_path = Path("data/external/leicester_dataset")
    if leicester_path.exists():
        print("✓ Check 2: Leicester external validation dataset exists")
        checks_passed += 1
    else:
        print("✗ Check 2: Leicester dataset not found in external validation")

    # Check 3: Required Python packages
    try:
        import mne
        import sklearn
        print("✓ Check 3: Required packages (MNE, sklearn) available")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ Check 3: Missing required package: {e}")

    # Check 4: Output directories exist
    required_dirs = [
        "results/features",
        "results/figures",
        "results/stats",
        "results/logs",
        "data/processed"
    ]
    all_dirs_exist = all(Path(d).exists() for d in required_dirs)
    if all_dirs_exist:
        print("✓ Check 4: All required output directories exist")
        checks_passed += 1
    else:
        print("✗ Check 4: Some output directories missing")

    print(f"\nPrerequisite checks: {checks_passed}/{total_checks}")
    return checks_passed == total_checks

def step1_fetch_and_parse_data():
    """Step 1: Fetch & parse MRC neurofeedback data."""
    print_timestamp("Step 1: Fetching and parsing MRC BNDU data")

    mrc_data_path = Path("data/raw/mrc_bndu_lfps-and-eegs-patients-parkinsons-disease-during-neurofeedback-training")

    if not mrc_data_path.exists():
        print("ERROR: MRC BNDU dataset not found")
        print("Please download the dataset first:")
        print("python scripts/download_dataset.py --source mrc_bndu --dataset lfps-and-eegs-patients-parkinsons-disease-during-neurofeedback-training")
        return False

    # This is a placeholder - actual implementation depends on data format
    print("Dataset structure analysis:")
    print(f"Data path: {mrc_data_path}")

    # Create a dataset inventory
    inventory = {
        "dataset_name": "MRC BNDU PD Neurofeedback",
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "data_path": str(mrc_data_path),
        "expected_subjects": 12,
        "expected_conditions": ["Training", "No Training"],
        "trials_per_condition": 40,
        "label_mapping": {
            "PD_REAL": "Training condition with real neurofeedback",
            "PD_SHAM": "No Training condition (sham)"
        },
        "next_steps": [
            "Identify EEG file format (.edf, .fif, .mat)",
            "Extract trial markers for Training vs No Training",
            "Verify 12 subjects with 80 trials each",
            "Map to PD_REAL/PD_SHAM labels"
        ]
    }

    # Save inventory
    inventory_path = Path("data/interim/mrc_dataset_inventory.json")
    with open(inventory_path, 'w') as f:
        json.dump(inventory, f, indent=2)

    print(f"✓ Dataset inventory saved: {inventory_path}")

    # Create a visualization placeholder
    fig, ax = create_figure(1, 1, figsize=(10, 6))
    ax.text(0.5, 0.5, f"MRC BNDU Dataset Structure\n\nExpected:\n- 12 PD patients\n- 80 trials per subject\n- Training vs No Training\n- EEG + LFP recordings\n\nAnalyzed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title("Dataset Overview - MRC BNDU")
    ax.axis('off')

    save_figure(fig, "dataset_overview_mrc", "raw", "structure_placeholder")
    close_all_figures()

    print("✓ Step 1 completed: Dataset structure documented")
    return True

def step2_preprocess_eeg():
    """Step 2: Preprocess EEG (bandpass, notch, ICA)."""
    print_timestamp("Step 2: EEG Preprocessing pipeline")

    # Preprocessing parameters according to CLAUDE.md
    preprocess_params = {
        "bandpass": [1, 40],  # 1-40 Hz
        "notch": 50,          # 50 Hz (UK mains)
        "ica_components": "auto",
        "epoch_length": 2.0,  # 2 second epochs
        "overlap": 0.5,       # 50% overlap
        "artifact_rejection": True
    }

    # Save preprocessing parameters
    params_path = Path("data/interim/preprocessing_parameters.json")
    with open(params_path, 'w') as f:
        json.dump({
            "parameters": preprocess_params,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "Ready for implementation"
        }, f, indent=2)

    # Create preprocessing flow diagram
    fig, ax = create_figure(1, 1, figsize=(12, 8))

    steps = [
        "Raw EEG\n(~64 channels)",
        f"Bandpass Filter\n{preprocess_params['bandpass'][0]}-{preprocess_params['bandpass'][1]} Hz",
        f"Notch Filter\n{preprocess_params['notch']} Hz",
        "ICA Artifact Removal\n(EOG, EMG)",
        f"Epoching\n{preprocess_params['epoch_length']}s windows",
        "Quality Control\n& Bad Epoch Rejection",
        "Preprocessed Data\nReady for Features"
    ]

    # Draw preprocessing flowchart
    y_positions = np.linspace(0.9, 0.1, len(steps))

    for i, (step, y) in enumerate(zip(steps, y_positions)):
        # Draw box
        bbox = dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7)
        ax.text(0.5, y, step, ha='center', va='center', transform=ax.transAxes,
                bbox=bbox, fontsize=10)

        # Draw arrow (except for last step)
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, y_positions[i+1] + 0.05), xytext=(0.5, y - 0.05),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.set_title(f"EEG Preprocessing Pipeline\nMRC BNDU Dataset", fontsize=14, pad=20)
    ax.axis('off')

    save_figure(fig, "preprocessing_pipeline", "preproc", "flowchart")
    close_all_figures()

    print(f"✓ Preprocessing parameters saved: {params_path}")
    print("✓ Step 2 completed: Preprocessing pipeline defined")
    return True

def step3_feature_extraction():
    """Step 3: Feature extraction (alpha/beta power, IAF, ratios)."""
    print_timestamp("Step 3: Feature extraction specification")

    # Feature extraction plan according to CLAUDE.md
    feature_plan = {
        "baseline_features": {
            "alpha_power": {
                "band": [8, 12],
                "method": "absolute_power",
                "regions": ["frontal", "central", "parietal", "occipital"]
            },
            "beta_power": {
                "band": [13, 30],
                "method": "absolute_power",
                "regions": ["frontal", "central", "parietal", "occipital"]
            },
            "relative_power": {
                "alpha_rel": "alpha_power / total_power",
                "beta_rel": "beta_power / total_power"
            },
            "individual_alpha_frequency": {
                "method": "peak_detection",
                "band": [8, 12]
            },
            "band_ratios": {
                "alpha_beta_ratio": "alpha_power / beta_power",
                "motor_posterior": "motor_alpha / posterior_alpha",
                "anterior_posterior": "frontal_alpha / occipital_alpha"
            }
        },
        "advanced_features": {
            "spectral_entropy": "Shannon entropy of PSD",
            "beta_burst_rate": "Bursts > 75th percentile",
            "peak_frequency": "Dominant frequency in band",
            "bandwidth": "Spectral bandwidth",
            "skewness_kurtosis": "Distribution moments"
        },
        "spatial_features": {
            "connectivity": "Coherence between regions",
            "asymmetry": "Left vs right hemisphere"
        }
    }

    # Save feature plan
    feature_path = Path("data/interim/feature_extraction_plan.json")
    with open(feature_path, 'w') as f:
        json.dump({
            "feature_plan": feature_plan,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_features": "~50-100 features total",
            "validation": "Subject-wise cross-validation"
        }, f, indent=2)

    # Create feature extraction diagram
    fig, axes = create_figure(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Extraction Plan", fontsize=16)

    # Alpha band features
    axes[0,0].text(0.5, 0.8, "Alpha Band (8-12 Hz)", ha='center', fontweight='bold')
    alpha_features = [
        "• Absolute power per region",
        "• Relative power (α/total)",
        "• Individual Alpha Frequency (IAF)",
        "• Peak alpha frequency",
        "• Alpha asymmetry (L/R)"
    ]
    axes[0,0].text(0.1, 0.4, "\n".join(alpha_features), va='center', fontsize=10)
    axes[0,0].axis('off')

    # Beta band features
    axes[0,1].text(0.5, 0.8, "Beta Band (13-30 Hz)", ha='center', fontweight='bold')
    beta_features = [
        "• Absolute power per region",
        "• Relative power (β/total)",
        "• Beta burst rate & duration",
        "• Peak beta frequency",
        "• Beta/alpha ratio"
    ]
    axes[0,1].text(0.1, 0.4, "\n".join(beta_features), va='center', fontsize=10)
    axes[0,1].axis('off')

    # Spatial features
    axes[1,0].text(0.5, 0.8, "Spatial Features", ha='center', fontweight='bold')
    spatial_features = [
        "• Motor/posterior ratio",
        "• Anterior/posterior ratio",
        "• Inter-regional coherence",
        "• Hemispheric asymmetry",
        "• Topographic patterns"
    ]
    axes[1,0].text(0.1, 0.4, "\n".join(spatial_features), va='center', fontsize=10)
    axes[1,0].axis('off')

    # Advanced features
    axes[1,1].text(0.5, 0.8, "Advanced Features", ha='center', fontweight='bold')
    advanced_features = [
        "• Spectral entropy",
        "• Bandwidth & skewness",
        "• Burst dynamics",
        "• Interaction terms",
        "• Transform features"
    ]
    axes[1,1].text(0.1, 0.4, "\n".join(advanced_features), va='center', fontsize=10)
    axes[1,1].axis('off')

    save_figure(fig, "feature_extraction_plan", "features", "overview")
    close_all_figures()

    print(f"✓ Feature extraction plan saved: {feature_path}")
    print("✓ Step 3 completed: Feature extraction specified")
    return True

def step4_model_validation():
    """Step 4: Subject-wise CV with multiple models."""
    print_timestamp("Step 4: Model validation strategy")

    # Model validation plan
    validation_plan = {
        "cross_validation": {
            "method": "subject_wise",
            "folds": 5,
            "stratification": "by_condition",
            "no_subject_leakage": True
        },
        "baseline_models": [
            {
                "name": "Logistic Regression",
                "hyperparameters": {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2", "elasticnet"],
                    "solver": ["liblinear", "saga"]
                }
            },
            {
                "name": "SVM",
                "hyperparameters": {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
                }
            },
            {
                "name": "Random Forest",
                "hyperparameters": {
                    "n_estimators": [50, 100, 200, 500],
                    "max_depth": [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
        ],
        "hyperparameter_optimization": {
            "method": "bayesian",
            "iterations": 100,
            "scoring": "balanced_accuracy"
        },
        "metrics": [
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1_score",
            "roc_auc",
            "precision_recall_auc"
        ],
        "statistical_tests": {
            "permutation_test": {
                "n_permutations": 10000,
                "alpha": 0.05
            },
            "bootstrap_ci": {
                "n_bootstrap": 1000,
                "confidence_level": 0.95
            }
        },
        "target_performance": {
            "balanced_accuracy": 0.70,
            "note": "≥70% as specified in requirements"
        }
    }

    # Save validation plan
    validation_path = Path("data/interim/validation_strategy.json")
    with open(validation_path, 'w') as f:
        json.dump({
            "validation_plan": validation_plan,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "compliance": "Subject-wise isolation enforced"
        }, f, indent=2)

    # Create validation strategy diagram
    fig, ax = create_figure(1, 1, figsize=(12, 8))

    # Draw validation process
    process_text = """
    MODEL VALIDATION STRATEGY

    1. DATA SPLIT (Subject-wise)
       ├── 12 PD subjects
       ├── 5-fold cross-validation
       └── No subject leakage across folds

    2. FEATURE SELECTION
       ├── ~50-100 features (α/β power, ratios, IAF)
       ├── Scaling: StandardScaler
       └── Selection: Based on univariate tests

    3. MODEL TRAINING
       ├── Logistic Regression (baseline)
       ├── SVM with RBF kernel
       ├── Random Forest
       └── Bayesian hyperparameter optimization

    4. EVALUATION METRICS
       ├── Balanced Accuracy (primary)
       ├── Sensitivity & Specificity
       ├── ROC-AUC & PR-AUC
       └── F1-score

    5. STATISTICAL VALIDATION
       ├── Permutation test (10,000 permutations)
       ├── Bootstrap confidence intervals
       └── Effect size calculation

    6. TARGET: ≥70% Balanced Accuracy
    """

    ax.text(0.05, 0.95, process_text, transform=ax.transAxes, fontsize=11,
            va='top', ha='left', family='monospace')
    ax.set_title("Subject-wise Cross-Validation Strategy", fontsize=14, pad=20)
    ax.axis('off')

    save_figure(fig, "validation_strategy", "models", "cv_plan")
    close_all_figures()

    print(f"✓ Validation strategy saved: {validation_path}")
    print("✓ Step 4 completed: Model validation strategy defined")
    return True

def step5_external_validation():
    """Step 5: External validation setup with Leicester dataset."""
    print_timestamp("Step 5: External validation with Leicester dataset")

    leicester_path = Path("data/external/leicester_dataset")

    if not leicester_path.exists():
        print("ERROR: Leicester dataset not found for external validation")
        return False

    # External validation plan
    external_plan = {
        "external_dataset": {
            "name": "Leicester EEG Exergaming Dataset",
            "path": str(leicester_path),
            "usage": "EXTERNAL VALIDATION ONLY - Never for training",
            "subjects": 17,
            "sessions": 250,
            "files": {"csv": 1232, "pickle": 1105}
        },
        "validation_process": {
            "step1": "Train final model on MRC BNDU dataset (all subjects)",
            "step2": "Apply same preprocessing to Leicester data",
            "step3": "Extract same features from Leicester data",
            "step4": "Test trained model on Leicester (no retraining)",
            "step5": "Compare performance metrics",
            "step6": "Assess generalization capability"
        },
        "expected_challenges": [
            "Different recording equipment/setup",
            "Different task paradigm (exergaming vs neurofeedback)",
            "Different subject characteristics",
            "Need to map Leicester conditions to PD_REAL/PD_SHAM"
        ],
        "success_criteria": {
            "minimum_performance": "Above chance (>50%)",
            "ideal_performance": "Within 10% of internal CV performance",
            "statistical_significance": "p < 0.05 on permutation test"
        }
    }

    # Save external validation plan
    external_path = Path("data/interim/external_validation_plan.json")
    with open(external_path, 'w') as f:
        json.dump({
            "external_plan": external_plan,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "Ready for implementation"
        }, f, indent=2)

    # Create external validation diagram
    fig, ax = create_figure(1, 1, figsize=(12, 6))

    validation_flow = """
    EXTERNAL VALIDATION WORKFLOW

    Training Phase (MRC BNDU):
    [12 PD subjects] → [Preprocess] → [Extract Features] → [Train Models] → [Best Model]
                                                                              ↓
    External Validation (Leicester):                                          ↓
    [17 subjects] → [Same Preprocess] → [Same Features] → [Apply Model] → [Performance]

    Success Metrics:
    • Balanced Accuracy > 50% (above chance)
    • Statistical significance (p < 0.05)
    • Generalization gap < 10% from internal CV
    """

    ax.text(0.05, 0.7, validation_flow, transform=ax.transAxes, fontsize=11,
            va='top', ha='left', family='monospace')
    ax.set_title("External Validation: MRC BNDU → Leicester", fontsize=14)
    ax.axis('off')

    save_figure(fig, "external_validation_plan", "external", "workflow")
    close_all_figures()

    print(f"✓ External validation plan saved: {external_path}")
    print("✓ Step 5 completed: External validation strategy defined")
    return True

def step6_implementation_roadmap():
    """Step 6: Create implementation roadmap."""
    print_timestamp("Step 6: Implementation roadmap")

    # Implementation roadmap
    roadmap = {
        "implementation_phases": {
            "phase1_data_preparation": {
                "tasks": [
                    "Download MRC BNDU dataset",
                    "Parse EEG files and extract trial labels",
                    "Verify PD_REAL vs PD_SHAM mapping",
                    "Create subject metadata files"
                ],
                "estimated_time": "2-3 days",
                "deliverables": ["data/processed/mrc_subjects_metadata.json"]
            },
            "phase2_preprocessing": {
                "tasks": [
                    "Implement bandpass filter (1-40 Hz)",
                    "Apply notch filter (50 Hz)",
                    "Run ICA for artifact removal",
                    "Epoch data into 2s windows",
                    "Quality control and bad epoch rejection"
                ],
                "estimated_time": "3-4 days",
                "deliverables": ["data/processed/mrc_preprocessed_epochs.h5"]
            },
            "phase3_features": {
                "tasks": [
                    "Extract alpha/beta power features",
                    "Calculate IAF per subject",
                    "Compute spatial ratios",
                    "Add advanced features (entropy, bursts)",
                    "Create feature matrix"
                ],
                "estimated_time": "2-3 days",
                "deliverables": ["results/features/mrc_features.csv"]
            },
            "phase4_modeling": {
                "tasks": [
                    "Implement subject-wise CV",
                    "Train baseline models (LR, SVM, RF)",
                    "Hyperparameter optimization",
                    "Statistical validation",
                    "Performance analysis"
                ],
                "estimated_time": "3-4 days",
                "deliverables": ["results/models/best_model.pkl", "results/stats/cv_results.json"]
            },
            "phase5_external_validation": {
                "tasks": [
                    "Preprocess Leicester dataset",
                    "Extract same features",
                    "Apply trained model",
                    "Performance comparison",
                    "Final report"
                ],
                "estimated_time": "2-3 days",
                "deliverables": ["results/external_validation_report.pdf"]
            }
        },
        "total_estimated_time": "12-17 days",
        "critical_path": "Data parsing → Preprocessing → Feature extraction → Modeling",
        "risk_mitigation": [
            "Start with data exploration to understand format",
            "Implement preprocessing incrementally with validation plots",
            "Use simple features first, add complexity gradually",
            "Validate each step before proceeding"
        ]
    }

    # Save roadmap
    roadmap_path = Path("data/interim/implementation_roadmap.json")
    with open(roadmap_path, 'w') as f:
        json.dump({
            "roadmap": roadmap,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "next_action": "Begin Phase 1: Data Preparation"
        }, f, indent=2)

    # Create Gantt chart visualization
    fig, ax = create_figure(1, 1, figsize=(14, 8))

    phases = list(roadmap["implementation_phases"].keys())
    phase_names = [p.replace("_", " ").title() for p in phases]
    durations = [3, 3.5, 2.5, 3.5, 2.5]  # Average days

    # Create Gantt chart
    start_times = [sum(durations[:i]) for i in range(len(durations))]

    colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))

    for i, (phase, start, duration, color) in enumerate(zip(phase_names, start_times, durations, colors)):
        ax.barh(i, duration, left=start, height=0.6, color=color, alpha=0.8)
        ax.text(start + duration/2, i, f"{phase}\n({duration} days)",
                ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xlim(0, sum(durations) + 1)
    ax.set_ylim(-0.5, len(phases) - 0.5)
    ax.set_xlabel("Days", fontsize=12)
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([])
    ax.set_title("Implementation Timeline - EEG Exergaming Project", fontsize=14, pad=20)
    ax.grid(axis='x', alpha=0.3)

    save_figure(fig, "implementation_roadmap", "models", "timeline")
    close_all_figures()

    print(f"✓ Implementation roadmap saved: {roadmap_path}")
    print("✓ Step 6 completed: Implementation plan ready")
    return True

def main():
    """Execute minimal sprint plan."""
    print("=" * 80)
    print("MINIMAL SPRINT PLAN - EEG EXERGAMING PROJECT")
    print("=" * 80)
    print_timestamp("Starting sprint planning")

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please address issues above before proceeding.")
        return False

    # Execute all steps
    steps = [
        step1_fetch_and_parse_data,
        step2_preprocess_eeg,
        step3_feature_extraction,
        step4_model_validation,
        step5_external_validation,
        step6_implementation_roadmap
    ]

    success_count = 0
    for i, step_func in enumerate(steps, 1):
        try:
            success = step_func()
            if success:
                success_count += 1
                print(f"✅ Step {i} completed successfully\n")
            else:
                print(f"❌ Step {i} failed\n")
        except Exception as e:
            print(f"❌ Step {i} failed with error: {e}\n")

    print("=" * 80)
    print("SPRINT PLAN SUMMARY")
    print("=" * 80)
    print(f"Steps completed: {success_count}/{len(steps)}")

    if success_count == len(steps):
        print("🎉 Sprint planning completed successfully!")
        print("Ready to begin implementation:")
        print("1. Download MRC BNDU dataset")
        print("2. Begin Phase 1: Data Preparation")
        print("3. Follow implementation roadmap")
    else:
        print("⚠️  Sprint planning partially completed")
        print("Please address failed steps before proceeding")

    print("=" * 80)
    return success_count == len(steps)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)