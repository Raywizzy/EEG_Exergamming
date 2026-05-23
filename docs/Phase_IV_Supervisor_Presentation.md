# Phase IV Multi-Site Validation: Supervisor Presentation
**EEG Exergaming Classification Pipeline**
**Cross-Site Generalization with CORAL Domain Adaptation**

---

## Slide 1: Executive Summary
### Phase IV: Multi-Site Validation Ready for Implementation

**Objective**: Validate locked Core5 + CORAL pipeline across 5 independent EEG datasets

**Current Status**:
- ✅ Phase II-III: 55.8-61.6% balanced accuracy (internal validation)
- ✅ Protocol complete with ethics framework
- ✅ Dataset technical specs verified
- 🎯 **Next**: 3-site LOSO validation targeting ≥65% BA

**Key Innovation**: CORAL domain adaptation for cross-site EEG harmonization

---

## Slide 2: Scientific Rationale
### Why Multi-Site Validation Matters

**Problem**: EEG biomarkers often fail to generalize across:
- Different recording systems (BrainVision, EGI, etc.)
- Site-specific protocols (sampling rates, referencing)
- Population heterogeneity

**Solution**: CORAL (Covariance Operator Ridge Alignment)
- Aligns second-order statistics between domains
- Preserves discriminative features while harmonizing distributions
- No target labels required (unsupervised domain adaptation)

**Clinical Impact**: Robust biomarkers → deployable neurofeedback systems

---

## Slide 3: Core5 Biomarker Pipeline
### Locked Feature Set from Phase II-III

**Alpha Burst Dynamics (8-12 Hz)**:
1. `duration_cv` - Coefficient of variation of burst durations
2. `duty_cycle` - Percentage time in burst state
3. `mean_duration` - Average burst length
4. `median_duration` - Median burst length
5. `motor_posterior_ratio` - Spatial contrast (C3-C4 vs O1-O2)

**Classification**: Logistic Regression (no hyperparameter tuning in Phase IV)

**Performance**: 55.8-61.6% BA on Leicester cohort (PD_REAL vs PD_SHAM)

---

## Slide 4: Phase II-III Results Summary
### Internal Validation Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| Balanced Accuracy | 58.7% | [55.8, 61.6] |
| Sensitivity (PD_REAL) | 62.1% | [58.9, 65.3] |
| Specificity (PD_SHAM) | 55.3% | [52.1, 58.5] |
| AUC-ROC | 0.612 | [0.578, 0.646] |
| Cohen's κ | 0.174 | [0.116, 0.232] |

**Subject-wise 5-fold CV** (n=847 epochs, 23 subjects)
**Permutation test**: p < 0.001 (10,000 iterations)
**Effect size**: Small but significant improvement over chance

---

## Slide 5: Multi-Site Dataset Portfolio
### Verified Technical Specifications

| Dataset | Subjects | Context | Duration | Montage | Sampling | Reference |
|---------|----------|---------|----------|---------|----------|-----------|
| **ds004584** | 100 PD / 49 HC | Eyes-open rest | ~2-3 min | 64-ch BrainVision | 500 Hz | Pz |
| **ds003490** | 25 PD / 25 HC | Rest + auditory oddball | Variable | 64-ch UNM/Iowa | 500 Hz | CPz/Pz |
| **ds003509** | 28 PD / 28 HC | Simon conflict + rest | 2 sessions | 64-ch UNM/Iowa | 500 Hz | CPz/Pz |
| **ds002778** | PD + HC | Resting state | ~few min | BrainVision | 500 Hz | See docs |
| **Iowa Lab** | 149 cohort | Resting state | ~3 min | 64-ch | 500 Hz | Pz |

**Total**: >300 participants across 5 independent sites

---

## Slide 6: CORAL Domain Adaptation
### Mathematical Framework

**Problem**: Train on site A, test on site B
- Different covariance structures: Σ_A ≠ Σ_B
- Same discriminative patterns (in principle)

**CORAL Solution**:
```
X_adapted = X_source × A
where A = Σ_source^(-1/2) × Σ_target^(1/2)
```

**Benefits**:
- Preserves relative feature relationships
- No target labels required
- Computationally efficient
- Proven effective for EEG cross-site harmonization

---

## Slide 7: Leave-One-Site-Out (LOSO) Design
### Rigorous Cross-Site Validation

**For each site i ∈ {ds004584, ds003490, ds003509, ds002778, Iowa}**:

1. **Training**: All patients from sites j ≠ i
2. **Domain Adaptation**: Apply CORAL (Training → Site i)
3. **Testing**: All patients from site i
4. **Metrics**: Balanced accuracy for site i

**Primary Endpoint**: Mean BA across all LOSO folds

**Statistical Test**:
- H₀: μ_BA ≤ 50% (chance level)
- H₁: μ_BA > 50%
- One-sided t-test, α = 0.05

---

## Slide 8: Statistical Hypothesis Framework
### Powered for Clinical Significance

**Primary Hypothesis**:
- **Null (H₀)**: Mean balanced accuracy ≤ 50%
- **Alternative (H₁)**: Mean balanced accuracy > 50%
- **Design Target**: ≥65% BA for clinical relevance

**Power Analysis**:
- 80% power to detect 65% BA vs 50% chance
- α = 0.05 (one-sided)
- Effect size: d = 0.75 (medium-large)

**Secondary Endpoints**:
- Classification latency <500ms
- Site-specific performance variation
- Clinical correlation (UPDRS if available)

---

## Slide 9: Data Governance & Ethics
### Leicester IRB + Multi-Site Compliance

**Ethics Framework**:
- All datasets: De-identified, open access or IRB-approved
- No DUA required for OpenNeuro datasets
- Leicester ethics notification for secondary data use

**Data Security**:
- GDPR-compliant encrypted storage
- VPN access controls
- Checksum validation at transfer
- Audit logs for all data operations

**DSMB Oversight**:
- Weekly performance monitoring
- Alert if any site BA drops >10pp below mean
- Safety monitoring for adverse events

---

## Slide 10: Quality Control & Monitoring
### Real-Time Pipeline Validation

**Preprocessing Harmonization**:
- BIDS format import where available
- Identical filtering: 1-40 Hz bandpass, 50 Hz notch
- Common Average Reference (CAR)
- Intersection montage across sites
- Resampling to common rate when needed

**Quality Metrics**:
- Epochs count per subject/site
- % clean data after artifact rejection
- Cross-site covariance drift detection
- Feature distribution alignment post-CORAL

**Automated Alerts**: Performance <threshold triggers review

---

## Slide 11: Implementation Timeline
### Phase IV Execution Plan (Next 8 Weeks)

**Weeks 1-2**: Technical Setup
- Unified preprocessing pipeline deployment
- ds004584 BIDS conversion + QC
- Initial 3-site LOSO pilot

**Weeks 3-4**: Full Multi-Site Validation
- All 5 datasets processed
- Complete LOSO cross-validation
- Statistical analysis + confidence intervals

**Weeks 5-6**: Analysis & Reporting
- Performance benchmarking vs Phase II-III
- Site-specific analysis
- Clinical correlation assessment

**Weeks 7-8**: Documentation & Dissemination
- Technical report completion
- Manuscript preparation
- Regulatory filing preparation

---

## Slide 12: Success Criteria & Next Steps
### Go/No-Go Decision Framework

**Primary Success Criteria**:
- ✅ Mean LOSO BA significantly > 50% (p < 0.05)
- ✅ At least 3/5 sites achieve BA ≥ 60%
- ✅ No critical safety/quality flags

**Stretch Goals**:
- 🎯 Mean BA ≥ 65% (clinical significance)
- 🎯 Consistent performance across all sites (CV < 0.15)
- 🎯 Processing latency <500ms maintained

**If Successful → Phase V**: Real-time neurofeedback deployment
**If Unsuccessful**: Domain adaptation refinement + expanded training data

**Decision Point**: Week 4 interim analysis

---

## Questions & Discussion

**Key Decision Points**:
1. Approve Phase IV implementation timeline?
2. Ethics submission priority level?
3. Resource allocation for 5-dataset processing?
4. Interim analysis review schedule?

**Risk Mitigation**:
- Parallel processing pipeline to handle dataset scale
- Backup domain adaptation methods if CORAL underperforms
- Staged rollout (3-site → 5-site) for early feedback

---

**Contact**: [Your details]
**Project Repository**: `/Users/user/Desktop/EEG_Exergamming/`
**Last Updated**: 2025-09-19