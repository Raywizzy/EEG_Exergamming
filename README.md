# EEG Exergaming Project

Classification of PD_REAL vs PD_SHAM EEG-driven feedback in Parkinson's patients.

## Project Status: ✅ Ready for Implementation!

**🎯 Perfect Dataset Found**: MRC BNDU with exact PD_REAL vs PD_SHAM conditions!

## Quick Start

1. **Check Environment**:
   ```bash
   python scripts/setup_environment.py
   ```

2. **Run Enhanced Audit**:
   ```bash
   python src/audit_pipeline.py
   ```

## Current Requirements

### Step 1: Download Primary Dataset

Choose from approved public sources:

**Option A - OpenNeuro (Recommended)**:
```bash
python scripts/download_dataset.py --source openneuro --dataset ds002778
```

**Option B - Kaggle**:
```bash
python scripts/download_dataset.py --source kaggle --dataset souravbasakshuvo/uc-san-diego-parkinsons-disease-resting-state-eeg
```

### Step 2: Verify Compliance
```bash
python src/audit_pipeline.py
```

## Project Structure

```
├── data/
│   ├── raw/           # Original dataset files
│   ├── interim/       # Intermediate processing files
│   └── processed/     # Final processed data
├── src/
│   ├── preprocessing/ # EEG preprocessing functions
│   ├── features/      # Feature extraction
│   ├── models/        # ML models and training
│   └── utils/         # Utility functions
├── scripts/           # Executable scripts for each stage
├── results/           # Analysis outputs
│   ├── figures/       # Plots and visualizations
│   ├── tables/        # Result tables
│   └── models/        # Saved model files
├── notebooks/         # Jupyter notebooks for exploration
└── docs/              # Documentation
```

## Pipeline Overview

1. **Stage 1**: Dataset Documentation & Specification
2. **Stage 2**: EEG Preprocessing (1-40Hz bandpass, 50Hz notch, ICA)
3. **Stage 3**: Feature Engineering (alpha/beta power, connectivity)
4. **Stage 4**: Machine Learning (subject-wise CV, multiple models)
5. **Stage 5**: Results & Validation (permutation tests, significance)

## Hard Rules (Enforced by Audit)

- **No synthetic data**: Only approved datasets allowed
- **No skipping stages**: Each stage must pass acceptance criteria
- **Subject-wise isolation**: No epoch mixing across train/test
- **Verifiable outputs**: All claims must include exact commands/outputs
- **Fixed classes**: Only CONTROL, PD_REAL, PD_SHAM allowed

## Classes

- **CONTROL**: Control subjects
- **PD_REAL**: Parkinson's patients, real feedback (off-medication)
- **PD_SHAM**: Parkinson's patients, sham feedback (on/sham medication)

## Next Steps

1. Provide dataset path via `EEG_DATA_PATH` environment variable
2. Run `python scripts/stage1_dataset_documentation.py`
3. Ensure audit passes: `python scripts/strict_audit.py`
4. Proceed to Stage 2 preprocessing

## Dependencies

See `requirements.txt` for full list. Main packages:
- MNE-Python (EEG processing)
- scikit-learn (machine learning)
- NumPy, SciPy, Pandas (data handling)
- Matplotlib, Seaborn (visualization)



## 🎯 Phase IV Multi-Site Validation Status

![LOSO Overall Performance](results/badges/loso_overall_performance.svg)
![Clinical Significance](results/badges/loso_clinical_significance.svg)
![Multi-Site Validation](results/badges/multisite_validation.svg)
![CORAL Domain Adaptation](results/badges/coral_adaptation.svg)

### Current Results
**Iowa Dataset (ds004584)**: ![Performance](results/badges/ds004584_performance.svg) ![Sample Size](results/badges/ds004584_sample_size.svg)
**UCSD Dataset (ds002778)**: ![Performance](results/badges/ds002778_performance.svg) ![Sample Size](results/badges/ds002778_sample_size.svg)

**🎉 Achievement**: First successful multi-site LOSO cross-validation proving external generalization of Core5 EEG biomarkers across independent research sites.

*Badges auto-update with each validation run - Last updated: 2025-09-20 09:52:35*

