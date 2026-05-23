# Phase III Protocol: Multi-Site Validation Study
**EEG Biomarker Pipeline for Parkinson's Disease Classification**

---

## Study Overview

### Objective
Validate the Core5 + CORAL biomarker pipeline across multiple independent sites to establish regulatory-grade evidence for clinical deployment.

### Primary Hypothesis
The Core5 feature set with CORAL domain adaptation will achieve ≥60% balanced accuracy in leave-one-site-out (LOSO) cross-validation across 3+ independent EEG acquisition sites.

### Study Design
- **Type**: Retrospective multi-site validation study
- **Validation Strategy**: Leave-One-Site-Out (LOSO) cross-validation
- **Population**: Parkinson's disease patients and controls
- **Duration**: 12 weeks (0-3 months Phase III timeline)

---

## Study Endpoints

### Primary Endpoint
**Cross-site generalization performance**
- **Metric**: Balanced accuracy (BA) in LOSO validation
- **Target**: Mean BA ≥ 60% across all LOSO splits
- **Statistical Requirement**: p < 0.05 (permutation test)
- **Clinical Requirement**: 95% CI lower bound ≥ 55%

### Secondary Endpoints
1. **Sensitivity and Specificity**
   - Target: Both ≥ 60% per site
   - Rationale: Clinical utility requires balanced performance

2. **ROC-AUC Performance**
   - Target: ≥ 0.65 per site
   - Rationale: Discriminative ability assessment

3. **Site Consistency**
   - Target: No site >10% below aggregate mean
   - Target: All sites ≥55% BA minimum
   - Rationale: Ensure robust deployment across diverse settings

4. **CORAL Alignment Effectiveness**
   - Target: ≥5% improvement over baseline transfer
   - Metric: Comparison of CORAL vs. direct transfer performance

---

## Study Population

### Inclusion Criteria
**Parkinson's Disease Subjects:**
- Clinical diagnosis of idiopathic Parkinson's disease
- Hoehn & Yahr stage I-III
- Age 50-80 years
- Stable medication regimen

**Control Subjects:**
- Age-matched healthy controls (±5 years)
- No neurological or psychiatric disorders
- No medications affecting CNS function

### Exclusion Criteria
- Secondary parkinsonism or atypical parkinsonian syndromes
- Significant cognitive impairment (MMSE < 24)
- Active psychiatric disorders
- Contraindications to EEG recording
- Poor quality EEG data (>20% artifact rate)

### Data Sources
1. **Leicester Dataset** (Target site #1)
   - n ≈ 242 recordings, 31 subjects
   - OFF/ON medication paradigm
   - 500 Hz sampling, standard montage

2. **MRC BNDU Dataset** (Source site)
   - n ≈ 514 recordings, 31 subjects
   - SHAM/REAL stimulation paradigm
   - 500 Hz sampling, motor/posterior regions

3. **UC San Diego Dataset** (Target site #2) [Planned]
   - ds002778: Parkinson's disease EEG data
   - Rest/task paradigms
   - Standard 10-20 montage

---

## Methodology

### Core5 Feature Set
**Validated β-burst biomarkers from Phase II:**
1. `duration_cv` - Coefficient of variation of burst duration
2. `duty_cycle` - Proportion of time in burst state
3. `mean_duration_ms` - Average burst duration (milliseconds)
4. `median_duration_ms` - Median burst duration (milliseconds)
5. `motor_posterior_duty_ratio` - Spatial selectivity index

### CORAL Domain Adaptation
**Implementation:**
- Correlation Alignment (Sun & Saenko, 2016)
- Eigendecomposition-based transformation
- Regularization parameter: 1e-6
- Applied between training sites and test site

### Preprocessing Pipeline
**Standardized across all sites:**
1. Bandpass filter: 13-30 Hz (β frequency band)
2. Common average reference (CAR)
3. Artifact rejection: Visual inspection + ICA
4. Epoching: 2-second non-overlapping windows
5. β-burst detection: Hilbert envelope + adaptive threshold
6. Quality control: <20% artifact rate, ≥5 minutes data

### Statistical Analysis Plan

#### LOSO Cross-Validation
- **Splits**: Train on (n-1) sites, test on 1 held-out site
- **Models**: Logistic regression (primary), SVM (secondary)
- **Features**: Core5 with StandardScaler normalization
- **Domain Adaptation**: CORAL applied to each train→test transfer

#### Statistical Tests
1. **Permutation Test** (n=1000)
   - Null hypothesis: No difference from chance performance
   - Labels shuffled while preserving subject structure
   - Significance threshold: α = 0.05

2. **Bootstrap Confidence Intervals** (n=1000)
   - 95% CI for balanced accuracy
   - Bias-corrected and accelerated (BCa) method
   - Requirement: Lower bound ≥ 55%

3. **Effect Size Analysis**
   - Cohen's d vs. null hypothesis
   - Clinical significance assessment
   - Between-site variance analysis

---

## Quality Control & Risk Management

### Data Quality Assurance
**Pre-processing QC:**
- Signal quality assessment (SNR > 20 dB)
- Artifact rate monitoring (<20% rejection)
- Channel impedance verification (<5 kΩ)
- Recording duration validation (≥5 minutes)

**Feature Quality:**
- Core5 feature stability checks
- Outlier detection (>3 SD from median)
- Missing value assessment (<5% tolerance)
- Distribution normality testing

### Risk Mitigation
**Technical Risks:**
- Site-specific preprocessing differences
  - *Mitigation*: Standardized SOP, harmonization validation
- CORAL transformation failure
  - *Mitigation*: Regularization, fallback to direct transfer
- Model overfitting to source sites
  - *Mitigation*: Subject-wise validation, permutation testing

**Scientific Risks:**
- Insufficient sample size per site
  - *Mitigation*: Power analysis, effect size monitoring
- Site-specific confounders
  - *Mitigation*: Demographic matching, subgroup analysis
- Performance degradation vs. Phase II
  - *Mitigation*: Progressive evaluation, error analysis

---

## Acceptance Criteria

### Gate A: Harmonization (Week 2)
- [ ] All sites pass preprocessing QC
- [ ] Core5 features extractable from all datasets
- [ ] CORAL transformation stable (condition number <1e12)
- [ ] Distribution alignment confirmed (Wasserstein distance <50)

### Gate B: Primary Results (Week 6)
- [ ] LOSO mean BA ≥ 60%
- [ ] Statistical significance: p < 0.05
- [ ] 95% CI lower bound ≥ 55%
- [ ] No site BA < 55% (consistency requirement)

### Gate C: Regulatory Package (Week 10)
- [ ] Complete validation report with SOPs
- [ ] Error analysis and subgroup breakdowns
- [ ] Risk assessment and mitigation documentation
- [ ] Reproducibility verification (independent run)

---

## Deliverables

### Technical Deliverables
1. **LOSO Validation Results** (`results/phase3_LOSO/`)
   - Performance metrics per site and aggregate
   - Statistical validation (permutation tests, CIs)
   - ROC curves, calibration plots, confusion matrices

2. **Error Analysis Report** (`results/phase3_error_analysis/`)
   - Misclassification patterns
   - Subgroup analysis (age, sex, disease duration)
   - Site-specific failure modes

3. **CORAL Analysis** (`results/phase3_coral/`)
   - Domain adaptation effectiveness
   - Covariance alignment visualization
   - Transfer learning performance gains

### Regulatory Deliverables
1. **Phase III Validation Report** (`docs/regulatory/phase3_validation_report.pdf`)
   - Executive summary for regulatory review
   - Complete methodology and results
   - Risk-benefit analysis and clinical utility

2. **Standard Operating Procedures** (`docs/SOPs/`)
   - Site calibration protocol
   - Preprocessing and QC procedures
   - Feature extraction and model deployment

3. **Software Package** (`src/deployment/`)
   - Production-ready pipeline code
   - Configuration management
   - Quality control automation

---

## Timeline & Milestones

### Phase IIIA: Data Integration (Weeks 1-2)
- **M1.1**: UC San Diego dataset integration
- **M1.2**: Preprocessing harmonization validation
- **M1.3**: Core5 feature extraction verification

### Phase IIIB: LOSO Validation (Weeks 3-6)
- **M2.1**: LOSO splits and baseline evaluation
- **M2.2**: CORAL domain adaptation optimization
- **M2.3**: Statistical validation and significance testing

### Phase IIIC: Analysis & Documentation (Weeks 7-10)
- **M3.1**: Error analysis and failure mode investigation
- **M3.2**: Regulatory documentation preparation
- **M3.3**: Reproducibility verification

### Phase IIID: Optional Extensions (Weeks 11-12)
- **M4.1**: Deep learning pilot comparison
- **M4.2**: Additional site integration (if available)
- **M4.3**: Phase IV clinical trial design

---

## Success Criteria Summary

**Phase III will be considered successful if:**

1. ✅ **Primary endpoint achieved**: LOSO BA ≥ 60% (p < 0.05)
2. ✅ **Confidence interval met**: 95% CI lower bound ≥ 55%
3. ✅ **Site consistency**: All sites ≥ 55% BA, variance <10%
4. ✅ **Statistical rigor**: Permutation tests and bootstrap CIs support claims
5. ✅ **Regulatory readiness**: Complete documentation package for FDA submission

**Upon success, Phase III will provide:**
- Multi-site validation evidence for regulatory submission
- Deployment-ready biomarker pipeline
- Framework for Phase IV prospective clinical trial
- Foundation for commercial clinical decision support system

---

**Protocol Version**: 1.0
**Date**: September 2025
**Next Review**: Phase III completion (Week 12)