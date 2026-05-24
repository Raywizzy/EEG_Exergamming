# EEG Exergaming Project

Subject-wise EEG analysis workflows for Parkinson's disease exergaming and neurofeedback research.

This repository contains code, configuration, documentation, datasets, generated artifacts, and validation outputs for the EEG exergaming project. Large datasets and generated binary artifacts are stored with Git LFS.

## Repository Status

- GitHub visibility: private
- Default branch: `phase2_domain_adaptation`
- Large files: tracked with Git LFS
- Latest local audit run: `2026-05-24`
- Audit status: failing because later-stage checks are still pending, not because the repository upload failed

The strict project rules are intentionally conservative: no synthetic data, no fabricated results, no skipped stages, and subject-wise validation only.

## Data And Provenance

Approved data sources are documented in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Do not add synthetic, placeholder, or fabricated data to this project.

Tracked data/artifact areas include:

- `data/`
- `bids/`
- `results/`
- `artifacts/`
- `models/`
- `logs/`
- `clinical_audit_logs/`

These paths are tracked through Git LFS. Clone users must install Git LFS before expecting the large files to resolve correctly.

```bash
git lfs install
git lfs pull
```

## Quick Start

Create an environment with the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the environment check:

```bash
python scripts/setup_environment.py
```

Run the strict audit:

```bash
python scripts/strict_audit.py
```

The audit is expected to fail until all required later-stage validation criteria are complete.

## Project Layout

```text
.
├── bids/                    # BIDS-formatted datasets tracked with Git LFS
├── clinical_mvp/            # Clinical MVP service and deployment files
├── config/                  # Pipeline and validation configuration
├── data/                    # Raw/interim/external data tracked with Git LFS
├── docs/                    # Protocol, roadmap, data source, and analysis docs
├── manuscripts/             # Manuscript drafts and submission materials
├── models/                  # Model metadata and model artifacts
├── results/                 # Validation outputs, figures, reports, and badges
├── scripts/                 # Stage scripts and operational utilities
└── src/                     # Core project package
```

## Core Workflow

1. Confirm data provenance against `docs/DATA_SOURCES.md`.
2. Run dataset documentation and validation scripts.
3. Run preprocessing only after stage acceptance criteria pass.
4. Extract features using the configured feature set.
5. Evaluate with subject-wise splits only.
6. Run cross-dataset validation and permutation tests where significance is claimed.
7. Run `scripts/strict_audit.py` before reporting completion.

## Compliance Rules

- No synthetic, fake, placeholder, or fabricated datasets.
- No invented metrics or unsupported claims.
- No epoch-wise leakage across train/test folds.
- Validation must be subject-wise.
- Class labels must match the project rules exactly.
- Every reported metric must be traceable to a command, output, and artifact path.
- If required files or tools are missing, stop and report the exact error.

## Git LFS Notes

This repository contains many LFS objects. A normal clone without Git LFS may show pointer files instead of the actual datasets/artifacts.

Useful checks:

```bash
git lfs ls-files | wc -l
git lfs status
git status --short --branch
```

## Current Validation Badges

![LOSO Overall Performance](results/badges/loso_overall_performance.svg)
![Clinical Significance](results/badges/loso_clinical_significance.svg)
![Multi-Site Validation](results/badges/multisite_validation.svg)
![CORAL Domain Adaptation](results/badges/coral_adaptation.svg)

Badges are generated artifacts and should be interpreted alongside the audit outputs and validation reports in `results/`.
