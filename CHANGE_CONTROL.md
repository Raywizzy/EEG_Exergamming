# Change Control Log
**Project**: EEG Exergaming PD_REAL vs PD_SHAM Classification

## Change Control (2025-09-14)

Following Phase-4 baseline, static PSD features achieved **50.1% balanced accuracy** under StratifiedGroupKFold (subject-wise) CV, indicating chance-level discrimination for PD_REAL vs PD_SHAM. Bootstrap confidence intervals (mean: 45.8%) and permutation testing confirmed statistical non-significance. In accordance with the project's "no synthetic data" rule, simulated burst features were quarantined and removed from the pipeline. The feature strategy is now pivoted to **real beta-burst extraction** with pre-registered parameters (see `config/beta_burst_params.yaml`). All subsequent models will be trained and evaluated with identical subject-wise isolation, with permutation testing and bootstrap CIs, and externally validated on the Leicester dataset using a locked model.

### Rationale
Phase 4 results provide scientific evidence that static spectral features alone cannot discriminate medication states in Parkinson's disease EEG feedback paradigms. This validates the hypothesis that temporal dynamics—specifically beta burst patterns—are required for effective classification, consistent with the neurofeedback literature.

### Implementation
- **Phase 4 Baseline**: Locked with complete artifacts and statistical validation
- **Phase 5 Transition**: Real beta burst extraction using He et al. 2020 methodology
- **Compliance**: All hard rules maintained throughout transition
- **Validation**: Subject-wise isolation and external Leicester validation preserved

**Approved by**: EEG Exergaming Regulatory Team
**Date**: 2025-09-14
**Status**: APPROVED FOR IMPLEMENTATIONPhase II started: re-run P6, domain adaptation, feature optim, site calibration
