# Statistical Analysis Plan (SAP)
**Phase IV Multi-Site EEG Biomarker Validation Study**

---

## Document Information

**Protocol Title**: Prospective Multi-Site Validation of EEG Beta-Burst Biomarkers for Parkinson's Disease Assessment
**SAP Version**: 1.0
**Date**: 2025-09-15
**Principal Investigator**: [Name]
**Institution**: University of Leicester
**Study Phase**: Phase IV Clinical Validation

---

## 1. Study Objectives and Endpoints

### 1.1 Primary Objective
To validate the locked Core5 + CORAL EEG biomarker pipeline for real-time classification of authentic vs sham neurofeedback in Parkinson's disease patients across multiple clinical sites.

### 1.2 Primary Endpoint
**Balanced Accuracy ≥65%** for real vs sham EEG feedback classification using the locked Core5 biomarker pipeline.

**Statistical Hypothesis**:
- H₀: Balanced Accuracy ≤ 50% (chance performance)
- H₁: Balanced Accuracy > 65% (clinically meaningful performance)

### 1.3 Secondary Endpoints
1. **Sensitivity**: ≥65% (true positive rate for PD_REAL detection)
2. **Specificity**: ≥65% (true negative rate for PD_SHAM rejection)
3. **ROC-AUC**: ≥0.70 (area under receiver operating characteristic curve)
4. **Processing Latency**: <500ms (real-time feasibility)
5. **Cross-Site Consistency**: ≤5% balanced accuracy variance across sites
6. **Clinical Correlation**: Pearson correlation r ≥ 0.4 with UPDRS-III scores

---

## 2. Study Design and Validation Framework

### 2.1 Study Design
- **Type**: Prospective, multi-site, observational validation study
- **Validation Method**: Leave-One-Site-Out (LOSO) Cross-Validation
- **Blinding**: Single-blind (patients unaware of real vs sham feedback)
- **Randomization**: Block randomization of session order within subjects

### 2.2 Locked Pipeline Approach
**Critical Requirement**: NO model retraining, hyperparameter tuning, or feature selection during Phase IV validation.

**Pipeline Components**:
1. **Feature Extraction**: Core5 biomarkers (fixed algorithm)
2. **Domain Adaptation**: CORAL transformation (pre-computed matrix from Phase III)
3. **Classification**: Logistic regression (fixed coefficients from Phase III)

---

## 3. Sample Size and Power Analysis

### 3.1 Sample Size Calculation
**Primary Endpoint Power Analysis**:
- **Target Performance**: 65% balanced accuracy
- **Null Hypothesis**: 50% (chance performance)
- **Alternative Hypothesis**: 65% (clinically meaningful)
- **Statistical Power**: 80%
- **Type I Error (α)**: 0.05 (two-sided)
- **Test**: One-sample t-test of balanced accuracy vs 50%

**Calculated Sample Size**: 80 patients (allowing 25% dropout → 100 recruited)

**Per-Site Distribution**:
- University of Leicester: 35 patients
- MRC BNDU Oxford: 35 patients
- UC San Diego: 30 patients
- **Total Target**: 100 patients

### 3.2 Power Analysis Assumptions
- **Expected Effect Size**: Cohen's d = 0.5 (medium effect)
- **Standard Deviation**: 15% (based on Phase III variability)
- **Correlation Structure**: Accounting for within-subject correlation
- **Missing Data**: 20% allowance for technical failures/dropouts

---

## 4. Analysis Populations

### 4.1 Analysis Sets Definition

#### Full Analysis Set (FAS)
**Definition**: All randomized patients who complete at least one EEG session
**Primary Analysis**: Yes - Primary endpoint analysis performed on FAS
**Missing Data Handling**: Multiple imputation for incomplete sessions

#### Per-Protocol Set (PPS)
**Definition**: All patients completing both EEG conditions (PD_REAL + PD_SHAM) with protocol adherence
**Secondary Analysis**: Sensitivity analysis of primary endpoint
**Protocol Deviations**: Major deviations leading to exclusion documented

#### Safety Analysis Set (SAS)
**Definition**: All patients undergoing any study procedure
**Purpose**: Safety reporting and adverse event analysis

### 4.2 Inclusion/Exclusion for Analysis

**Analysis Inclusion**:
- ✅ Complete EEG sessions for both conditions
- ✅ Signal quality meeting QC thresholds (≥80% epochs valid)
- ✅ No major protocol deviations affecting data integrity

**Analysis Exclusion**:
- ❌ <50% valid epochs due to artifacts
- ❌ Technical failures preventing classification
- ❌ Protocol violations affecting biomarker extraction

---

## 5. Statistical Methods

### 5.1 Primary Analysis Method

#### Leave-One-Site-Out (LOSO) Cross-Validation
```
For each site i ∈ {Leicester, BNDU, UCSD}:
    1. Training Set = All patients from sites j ≠ i
    2. Test Set = All patients from site i
    3. Apply CORAL domain adaptation: Training → Test site
    4. Classify test patients using locked Core5 + LR pipeline
    5. Calculate balanced accuracy for site i

Primary Endpoint = Mean(BA_Leicester, BA_BNDU, BA_UCSD)
```

#### Statistical Test for Primary Endpoint
**Method**: One-sample t-test of site-specific balanced accuracies
**Null Hypothesis**: μ_BA ≤ 50%
**Alternative Hypothesis**: μ_BA > 65%
**Significance Level**: α = 0.05 (one-sided test)
**Confidence Interval**: 95% CI for mean balanced accuracy

#### Handling of Multiple Sites
**Fixed Effects Model**: Each site treated as independent validation
**Random Effects Model**: Site effects modeled as random (sensitivity analysis)
**Heterogeneity Assessment**: Cochran's Q test for site consistency

### 5.2 Secondary Analysis Methods

#### Performance Metrics
**Sensitivity**: TP / (TP + FN) - True positive rate
**Specificity**: TN / (TN + FP) - True negative rate
**Positive Predictive Value**: TP / (TP + FP) - Precision
**Negative Predictive Value**: TN / (TN + FN) - Negative precision
**F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

#### ROC Analysis
**ROC Curve**: Generated using prediction probabilities
**AUC Calculation**: Trapezoidal rule integration
**AUC Confidence Interval**: DeLong method for correlated ROC curves
**Optimal Threshold**: Youden Index maximization

#### Cross-Site Consistency Analysis
**Method**: Analysis of Variance (ANOVA) with site as factor
**Metric**: Coefficient of variation across site-specific balanced accuracies
**Acceptance Criterion**: CV ≤ 5%
**Post-hoc Tests**: Tukey HSD for pairwise site comparisons

### 5.3 Clinical Correlation Analysis

#### UPDRS-III Correlation
**Method**: Pearson correlation between biomarker values and UPDRS-III scores
**Hypothesis**: r ≥ 0.4 (moderate clinical correlation)
**Confidence Interval**: 95% CI using Fisher's z-transformation
**Subgroup Analysis**: ON vs OFF medication states

#### Medication Effect Analysis
**Design**: Paired t-test comparing biomarkers ON vs OFF dopaminergic therapy
**Effect Size**: Cohen's d for within-subject medication effect
**Clinical Relevance**: ≥10% biomarker change with medication

---

## 6. Missing Data Handling

### 6.1 Missing Data Patterns
**Expected Missing Data**:
- Technical failures: 5-10% of sessions
- Patient dropouts: 15-20% of enrolled patients
- Protocol deviations: 5% of sessions

### 6.2 Missing Data Methods

#### Multiple Imputation (MI)
**Method**: Fully conditional specification (FCS)
**Imputations**: m = 20 imputed datasets
**Pooling**: Rubin's rules for combining results
**Auxiliary Variables**: Demographics, baseline UPDRS, site indicators

#### Sensitivity Analyses
1. **Complete Case Analysis**: Only patients with complete data
2. **Worst Case Imputation**: Missing classified as incorrect
3. **Best Case Imputation**: Missing classified as correct
4. **Pattern Mixture Models**: Different models by missing data pattern

---

## 7. Interim Analysis Plan

### 7.1 Interim Analysis Timing
**Analysis Point**: After 50% enrollment (n = 50 patients)
**Purpose**: Futility assessment and sample size re-estimation
**Blinding**: Maintained during interim analysis

### 7.2 Interim Analysis Procedures

#### Futility Boundary
**Method**: Conditional power calculation
**Futility Threshold**: Conditional power < 20%
**Decision Rule**: Stop for futility if P(Success | Observed Data) < 0.20

#### Sample Size Re-estimation
**Method**: Blinded sample size re-estimation based on observed variance
**Adjustment**: Up to 50% increase in sample size if needed
**Approval**: Protocol amendment required for sample size changes

### 7.3 Data Safety Monitoring Board (DSMB)
**Composition**: Independent statistician, clinical expert, regulatory expert
**Responsibilities**: Safety monitoring, futility assessment, protocol recommendations
**Meetings**: Quarterly during enrollment period

---

## 8. Handling of Protocol Deviations

### 8.1 Protocol Deviation Classification

#### Major Deviations (Exclusion from PPS)
- Wrong EEG acquisition parameters
- <50% valid epochs due to preventable technical issues
- Incorrect randomization or session order
- Use of excluded medications during sessions

#### Minor Deviations (Included in PPS with documentation)
- Small variations in session timing
- Minor electrode placement variations within standards
- Brief interruptions not affecting data quality

### 8.2 Protocol Deviation Reporting
**Documentation**: All deviations recorded in eCRF with impact assessment
**Review Process**: Medical monitor review within 48 hours
**Corrective Actions**: Protocol amendments or additional training as needed

---

## 9. Statistical Software and Reproducibility

### 9.1 Statistical Software
**Primary Software**: R version 4.3+ with validated packages
**Key Packages**:
- `caret` for machine learning validation
- `pROC` for ROC analysis
- `lme4` for mixed effects models
- `mice` for multiple imputation

### 9.2 Reproducibility Requirements
**Version Control**: Git repository for all analysis code
**Seed Setting**: Random seed = 42 for all analyses
**Documentation**: Comprehensive code documentation and comments
**Validation**: Independent statistical review of all code

---

## 10. Data Quality Control

### 10.1 Real-Time Quality Control
**Signal Quality Monitoring**:
- Electrode impedance < 10 kΩ continuous monitoring
- Automated artifact detection with real-time feedback
- Signal-to-noise ratio assessment per electrode
- Movement artifact detection and flagging

### 10.2 Post-Processing Quality Control
**Data Validation Checks**:
- EEG signal integrity verification
- Beta-burst detection validation
- Feature extraction range checks (physiological plausibility)
- Statistical outlier detection (mean ± 3 SD)

**Quality Control Reports**:
- Weekly site-specific QC dashboards
- Monthly cross-site comparison reports
- Real-time alerts for critical quality issues

---

## 11. Reporting and Documentation

### 11.1 Statistical Report Structure
1. **Executive Summary**: Primary results and conclusions
2. **Study Design**: Objectives, endpoints, methodology
3. **Analysis Populations**: FAS, PPS, safety populations
4. **Demographics**: Baseline characteristics by site
5. **Primary Analysis**: LOSO validation results
6. **Secondary Analysis**: Performance metrics and correlations
7. **Safety Analysis**: Adverse events and technical issues
8. **Conclusions**: Clinical and regulatory implications

### 11.2 Tables and Figures Plan
**Table 1**: Demographics and baseline characteristics
**Table 2**: Primary endpoint results by site
**Table 3**: Secondary endpoints summary
**Table 4**: Clinical correlation analysis
**Table 5**: Protocol deviations and missing data

**Figure 1**: LOSO validation performance by site
**Figure 2**: ROC curves for each site
**Figure 3**: Clinical correlation scatterplots
**Figure 4**: Biomarker distribution by medication state

---

## 12. Timeline and Deliverables

### 12.1 Analysis Timeline
**Database Lock**: Month 15 (end of data collection)
**Primary Analysis**: Month 16 (4 weeks post-lock)
**Secondary Analysis**: Month 17 (8 weeks post-lock)
**Final Report**: Month 18 (12 weeks post-lock)

### 12.2 Statistical Deliverables
- **SAP Finalization**: Before first patient enrollment
- **Interim Analysis Report**: Month 10
- **Primary Analysis Results**: Month 16
- **Final Statistical Report**: Month 18
- **Regulatory Submission Package**: Month 18

---

**SAP Approval**:
- [ ] Principal Investigator Approval
- [ ] Biostatistician Approval
- [ ] DSMB Approval
- [ ] Regulatory Affairs Review

**SAP Status**: Ready for implementation and regulatory submission
**Next Review**: Post-enrollment completion for any protocol amendments