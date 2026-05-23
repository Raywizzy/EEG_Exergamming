# Phase IV Clinical Protocol
**Prospective Multi-Site Validation of EEG Biomarkers for Parkinson's Disease Assessment**

---

## Study Information

**Protocol Title**: Prospective Multi-Site Validation of EEG Beta-Burst Biomarkers for Real-Time Parkinson's Disease Assessment in Exergaming Feedback Classification

**Protocol Version**: 1.0
**Date**: 2025-09-15
**Principal Investigator**: [Your Name]
**Institution**: University of Leicester
**Study Type**: Prospective, multi-site, observational validation trial
**Registration**: [To be registered with ClinicalTrials.gov]

---

## 1. Abstract

### Background
Parkinson's Disease (PD) affects >10 million people worldwide, with current clinical assessments relying on subjective scales and expensive imaging. Our validated EEG beta-burst biomarkers demonstrated 55.8% balanced accuracy in multi-site Leave-One-Site-Out validation (Phase III), with Core5 feature optimization achieving 61.6% within-site performance.

### Objective
To prospectively validate the locked Core5 + CORAL EEG biomarker pipeline for distinguishing real vs sham neurofeedback in PD patients across multiple clinical sites.

### Design
Prospective, multi-site validation study with locked machine learning pipeline and real-time EEG processing.

### Setting
3+ clinical sites: University of Leicester, MRC BNDU Oxford, UC San Diego, potential additional European centers.

### Participants
80-100 PD patients aged 40-80 years, capable of performing exergaming tasks.

### Primary Outcome
Balanced accuracy ≥65% for real vs sham EEG feedback classification using locked Core5 biomarkers.

---

## 2. Background and Rationale

### 2.1 Clinical Need
- **Current Limitations**: PD assessment relies on subjective UPDRS scales, expensive DaTscans, or invasive procedures
- **Unmet Need**: Objective, accessible, real-time biomarkers for PD motor state assessment
- **Market Opportunity**: $2.8B neurological assessment market with growing demand for objective tools

### 2.2 Scientific Foundation
**Phase III Achievements**:
- ✅ Multi-site validation framework established (Leicester + BNDU + UCSD)
- ✅ Core5 biomarker optimization: 61.6% BA within-dataset validation
- ✅ LOSO cross-dataset validation: 55.8% BA with statistical significance
- ✅ Regulatory-grade statistical validation methodology

**Core5 Biomarkers (Locked Pipeline)**:
1. `duration_cv`: Coefficient of variation of β-burst duration
2. `duty_cycle`: Proportion of time in β-burst state
3. `mean_duration_ms`: Average burst duration
4. `median_duration_ms`: Median burst duration
5. `motor_posterior_duty_ratio`: Spatial ratio between motor and posterior regions

### 2.3 Regulatory Positioning
- **FDA Pathway**: 510(k) predicate device pathway (EEG analysis software)
- **EU Pathway**: MDR Class IIa Software as Medical Device (SaMD)
- **Clinical Evidence**: Phase IV provides prospective multi-site validation required for submission

---

## 3. Objectives

### 3.1 Primary Objective
To validate the locked Core5 + CORAL EEG biomarker pipeline for real-time classification of authentic vs sham neurofeedback in Parkinson's disease patients across multiple clinical sites.

### 3.2 Secondary Objectives
1. **Cross-Site Generalization**: Evaluate biomarker consistency across ≥3 independent clinical centers
2. **Real-Time Feasibility**: Assess processing latency for clinical deployment (<500ms target)
3. **Clinical Correlation**: Measure biomarker correlation with established PD assessments (UPDRS-III, medication timing)
4. **Technical Validation**: Confirm locked pipeline performance without retraining or optimization
5. **Regulatory Evidence**: Generate prospective evidence package for FDA/CE submission

---

## 4. Primary and Secondary Endpoints

### 4.1 Primary Endpoint
**Primary Endpoint**: Balanced Accuracy (BA) of the locked Core5 + CORAL pipeline for real vs sham EEG feedback classification

**Design Target**: ≥65% BA in cross-site validation, consistent with clinically meaningful performance

**Statistical Hypothesis**:
- Null hypothesis (H₀): μ_BA ≤ 50% (chance level)
- Alternative hypothesis (H₁): μ_BA > 50%
- The design target of ≥65% BA represents the benchmark for prospective success but is not the formal hypothesis test threshold

**Justification**:
- Statistical testing vs chance (50%) provides rigorous validation framework
- 65% design target represents clinically meaningful improvement
- Achievable based on Phase III 55.8% baseline + targeted optimization
- Aligns with regulatory expectations for objective biomarker performance

### 4.2 Secondary Endpoints

#### Performance Metrics
- **Sensitivity**: ≥65% (PD_REAL detection rate)
- **Specificity**: ≥65% (PD_SHAM rejection rate)
- **ROC-AUC**: ≥0.70 (discrimination capability)
- **Positive Predictive Value**: ≥65% (clinical utility)

#### Technical Metrics
- **Processing Latency**: <500ms from EEG acquisition to classification
- **Cross-Site Consistency**: ≤5% balanced accuracy variance across sites
- **Signal Quality**: ≥80% epochs meeting quality control thresholds

#### Clinical Correlation
- **UPDRS-III Correlation**: r ≥ 0.4 with motor severity scores
- **Medication Effect**: Detectable biomarker changes with ON/OFF states
- **Clinical Utility**: Physician assessment of biomarker clinical relevance

---

## 5. Study Design and Setting

### 5.1 Study Design
- **Type**: Prospective, multi-site, observational validation trial
- **Blinding**: Single-blind (patients unaware of real vs sham feedback)
- **Randomization**: Block randomization of real/sham session order
- **Duration**: 18 months total (12 months recruitment + 6 months analysis)

### 5.2 Study Sites
**Primary Sites**:
1. **University of Leicester** (Lead site): Space Park Leicester, established EEG expertise
2. **MRC BNDU Oxford**: Neurofeedback training specialization
3. **UC San Diego**: Public dataset validation and US site representation

**Potential Additional Sites**:
- Partner neurology departments (2-3 additional sites)
- European collaborations through existing networks
- Industry partnerships (medtech validation)

### 5.3 Study Population
**Target Sample Size**: 80-100 PD patients
- **Per-Site Target**: 25-35 patients per primary site
- **Power Calculation**: 80% power to detect 65% vs 50% balanced accuracy (α=0.05)
- **Stratification**: Balanced by age, disease duration, medication status

---

## 6. Participants

### 6.1 Inclusion Criteria
1. **Clinical Diagnosis**: Parkinson's disease per UK Brain Bank criteria
2. **Age Range**: 40-80 years
3. **Motor Function**: Capable of performing simple exergaming tasks
4. **Medication Status**: Stable dopaminergic therapy (≥4 weeks)
5. **Consent**: Informed consent for EEG recording and data sharing
6. **Cognitive Function**: Mini-Mental State Exam ≥24

### 6.2 Exclusion Criteria
1. **Implanted Devices**: Deep brain stimulation electrodes (EEG interference)
2. **Neurological Comorbidities**: Stroke, epilepsy, significant head injury
3. **Psychiatric Conditions**: Active psychosis, severe depression (BDI >20)
4. **Technical Limitations**: Unable to tolerate EEG cap placement
5. **Medical Instability**: Acute illness requiring hospitalization
6. **Medication Changes**: Recent dopaminergic therapy adjustments (<4 weeks)

### 6.3 Withdrawal Criteria
- Patient request for withdrawal
- Development of exclusion criteria during study
- Protocol violations affecting data quality
- Adverse events related to study procedures

---

## 7. Interventions and Procedures

### 7.1 EEG Acquisition Protocol
**Hardware Specifications**:
- **EEG System**: 32-channel clinical EEG (Brain Products actiCHamp or equivalent)
- **Electrode Montage**: International 10-20 system with focus on motor/posterior regions
- **Sampling Rate**: 1000 Hz acquisition, decimated to 250 Hz for analysis
- **Reference**: Common average reference (CAR)

**Signal Processing Pipeline**:
1. **Preprocessing**: 1-40 Hz bandpass filter, 50 Hz notch filter
2. **Beta Extraction**: 13-30 Hz bandpass, Hilbert transform for envelope
3. **Burst Detection**: Amplitude threshold (75th percentile) + minimum duration (100ms)
4. **Feature Extraction**: Core5 biomarkers computed in real-time
5. **Classification**: Locked CORAL + Logistic Regression model

### 7.2 Exergaming Protocol
**Session Structure**:
- **Duration**: 15 minutes per condition (PD_REAL, PD_SHAM)
- **Rest Periods**: 5-minute breaks between conditions
- **Task**: Simple reaching/pointing exercises with visual feedback

**Feedback Conditions**:
- **PD_REAL**: Visual feedback directly linked to patient's beta-burst patterns
- **PD_SHAM**: Randomized visual feedback independent of EEG signals
- **Blinding**: Patients unaware of which condition is active

### 7.3 Clinical Assessments
**Baseline Assessment**:
- Demographics, medical history, current medications
- UPDRS-III motor examination
- Mini-Mental State Examination
- Beck Depression Inventory
- Hoehn & Yahr staging

**Session Assessments**:
- Medication timing (ON/OFF state)
- Fatigue scale (pre/post session)
- Patient experience questionnaire

---

## 8. Data Collection and Management

### 8.1 Data Types and Sources
**EEG Data**:
- Raw EEG signals (32 channels × 250 Hz × 15 minutes)
- Processed beta envelopes and burst detections
- Real-time Core5 biomarker extractions
- Classification outputs and confidence scores

**Clinical Data**:
- UPDRS-III scores and subscales
- Medication logs and timing
- Demographic and medical history
- Session questionnaires and observations

### 8.2 Data Quality Control
**Real-Time QC**:
- Electrode impedance monitoring (<10 kΩ)
- Artifact detection and epoch rejection
- Signal quality metrics and alerts
- Continuous recording verification

**Post-Processing QC**:
- Visual inspection of EEG signals
- Automated artifact detection algorithms
- Burst detection validation
- Feature extraction verification

### 8.3 Data Management
**Storage and Security**:
- Encrypted local storage with regular backups
- Pseudonymized patient identifiers
- GDPR-compliant data handling procedures
- Audit trail for all data access and modifications

**Data Sharing**:
- De-identified dataset for regulatory submission
- Open science data release (post-publication)
- Compliance with institutional data sharing policies

---

## 9. Locked Pipeline Validation

### 9.1 Pipeline Components (NO RETRAINING)
**Feature Extraction**: Core5 biomarkers (duration_cv, duty_cycle, mean_duration_ms, median_duration_ms, motor_posterior_duty_ratio)

**Domain Adaptation**: CORAL transformation matrix (pre-computed from Phase III)

**Classification Model**: Logistic regression with fixed coefficients from Phase III optimization

**Critical Requirement**: NO model retraining, hyperparameter tuning, or feature selection during Phase IV

### 9.2 Real-Time Implementation
**Latency Requirements**:
- Signal processing: <200ms
- Feature extraction: <200ms
- Classification: <100ms
- **Total latency**: <500ms for clinical utility

**Technical Validation**:
- Latency benchmarking on clinical hardware
- Stress testing with continuous operation
- Failure mode analysis and recovery procedures

---

## 10. Statistical Analysis Plan

### 10.1 Sample Size Calculation
**Primary Endpoint Power Analysis**:
- **Target Performance**: 65% balanced accuracy
- **Null Hypothesis**: 50% (chance performance)
- **Power**: 80%
- **Alpha**: 0.05 (two-sided)
- **Calculated Sample Size**: 80 patients (allowing 20% dropout → 100 recruited)

### 10.2 Analysis Populations
**Primary Analysis Population**: All patients completing both EEG conditions with quality control standards

**Per-Protocol Population**: Patients with full protocol compliance and complete data

**Safety Population**: All patients undergoing any study procedure

### 10.3 Statistical Methods
**Primary Analysis**:
- **Validation Framework**: Leave-One-Site-Out (LOSO) cross-validation
- **Metrics**: Balanced accuracy with 95% confidence intervals
- **Statistical Test**: Permutation testing (1000 iterations) for significance vs chance

**Secondary Analyses**:
- Sensitivity, specificity, ROC-AUC with bootstrap confidence intervals
- Cross-site consistency analysis (ANOVA)
- Clinical correlation analysis (Pearson/Spearman)
- Latency analysis (descriptive statistics)

### 10.4 Interim Analysis
**Planned Interim Analysis**: After 50% enrollment (40 patients)
- Safety review and protocol modifications if needed
- Futility analysis (conditional power <20%)
- Data quality assessment and site performance review

---

## 11. Ethics and Regulatory Considerations

### 11.1 Ethical Approval
**Institutional Review Boards**:
- University of Leicester Research Ethics Committee (lead approval)
- Local ethics committees at each participating site
- Harmonized protocol across all sites

**Informed Consent**:
- Written informed consent for all participants
- Specific consent for EEG recording and data sharing
- Right to withdraw without affecting clinical care

### 11.2 Data Protection
**GDPR Compliance**:
- Lawful basis: Scientific research with public interest
- Data minimization and purpose limitation
- Pseudonymization of all patient identifiers
- Secure data transfer between sites

**Patient Privacy**:
- De-identification procedures for all datasets
- Restricted access to identifiable information
- Data retention policies and destruction schedules

### 11.3 Regulatory Preparation
**FDA Engagement**:
- Pre-Submission (Q-Sub) meeting planning
- Software as Medical Device (SaMD) classification
- 510(k) predicate device identification

**EU Regulatory**:
- Medical Device Regulation (MDR) compliance
- Notified Body consultation for Class IIa device
- Clinical evidence requirements assessment

---

## 12. Risk Management and Safety

### 12.1 Risk Assessment
**Technical Risks**:
- EEG signal quality degradation → Real-time QC monitoring
- Processing latency exceeding targets → Hardware optimization
- Cross-site data variability → Standardized protocols

**Clinical Risks**:
- Patient fatigue during sessions → Regular breaks and monitoring
- Anxiety with EEG placement → Preparation and support
- Incidental EEG findings → Referral protocols established

**Regulatory Risks**:
- Performance below targets → Interim analysis with futility boundaries
- Data quality issues → Robust QC and validation procedures

### 12.2 Safety Monitoring
**Data Safety Monitoring Board**: Independent committee for safety oversight

**Adverse Event Reporting**: Systematic collection and reporting of any study-related adverse events

**Protocol Deviations**: Documentation and impact assessment of any protocol violations

---

## 13. Timeline and Milestones

### 13.1 Study Timeline (18 Months)

**Months 1-3: Setup Phase**
- Ethics approvals across all sites
- Protocol harmonization and training
- Equipment calibration and validation
- Staff training and certification

**Months 4-9: Recruitment Phase**
- Patient screening and enrollment
- Baseline assessments and randomization
- Initial data collection and QC

**Months 7-15: Data Collection Phase**
- EEG sessions across all sites
- Real-time biomarker validation
- Continuous quality monitoring
- Interim analysis (Month 10)

**Months 13-18: Analysis and Reporting**
- Centralized data analysis
- Statistical validation and reporting
- Manuscript preparation
- Regulatory submission package

### 13.2 Key Milestones
- **Month 3**: All sites activated and ready
- **Month 6**: 25% enrollment achieved
- **Month 10**: Interim analysis completed
- **Month 15**: Data collection completed
- **Month 18**: Final report and publication submission

---

## 14. Dissemination Plan

### 14.1 Primary Publications
**Target Journal**: The Lancet Digital Health
**Manuscript Title**: "Prospective Multi-Site Validation of EEG Biomarkers for Real-Time Parkinson's Disease Assessment"

### 14.2 Secondary Publications
- Technical validation in IEEE Transactions on Biomedical Engineering
- Clinical implementation in Movement Disorders
- Regulatory science perspective in Nature Medicine

### 14.3 Conference Presentations
- Organization for Human Brain Mapping (OHBM)
- Society for Neuroscience (SfN)
- IEEE Engineering in Medicine and Biology Conference (EMBC)
- International Conference on Movement Disorders (MDS)

### 14.4 Data Sharing
- De-identified dataset release via OpenNeuro
- Code availability via GitHub with DOI
- Regulatory submission documents (non-proprietary sections)

---

## 15. Budget and Resources

### 15.1 Personnel Requirements
**Per Site**:
- Clinical investigator (20% effort)
- Research coordinator (40% effort)
- EEG technician (30% effort)
- Data manager (10% effort)

**Central Coordination**:
- Principal investigator (50% effort)
- Biostatistician (25% effort)
- Data analyst (40% effort)
- Regulatory consultant (10% effort)

### 15.2 Equipment and Infrastructure
- EEG systems and maintenance contracts
- Secure data storage and transfer systems
- Real-time processing hardware
- Quality control and monitoring software

### 15.3 Funding Strategy
**Grant Applications**:
- NIH STTR/SBIR Phase II ($1.5M)
- Innovate UK Biomedical Catalyst ($500K)
- EU Horizon Europe Health ($750K)
- Industry partnerships and co-funding

---

## 16. Success Criteria and Go/No-Go Decisions

### 16.1 Primary Success Criteria
- ✅ Balanced accuracy ≥65% in primary endpoint analysis
- ✅ Statistical significance vs chance performance (p < 0.05)
- ✅ Real-time processing latency <500ms achieved
- ✅ Cross-site consistency within acceptable limits

### 16.2 Regulatory Readiness Criteria
- ✅ Complete prospective multi-site validation dataset
- ✅ Locked pipeline performance documentation
- ✅ Clinical utility evidence and physician validation
- ✅ Safety and usability assessment completed

### 16.3 Go/No-Go Decision Points
**Interim Analysis (Month 10)**:
- Conditional power ≥30% for primary endpoint
- Acceptable safety profile maintained
- Cross-site data quality standards met

**Study Completion (Month 18)**:
- Primary endpoint achieved → Proceed to regulatory submission
- Primary endpoint missed → Additional optimization or pivoting to research focus

---

## 17. Protocol Amendments and Version Control

### 17.1 Amendment Process
- Major amendments require ethics committee approval
- Minor amendments documented with version control
- All sites notified of protocol changes
- Amendment history maintained in study master file

### 17.2 Protocol Compliance
- Regular monitoring visits to each site
- Source data verification for critical data points
- Training updates for protocol changes
- Compliance metrics reporting

---

## Conclusion

Phase IV represents the critical translation of our validated EEG biomarker framework into prospective clinical evidence. Success will establish the first regulatory-ready EEG biomarker for Parkinson's disease assessment, opening pathways for clinical deployment and commercial translation.

**The locked Core5 + CORAL pipeline, validated through rigorous Phase III multi-site testing, is ready for prospective clinical validation that will transform objective PD assessment from research achievement to clinical reality.**

---

**Protocol Version**: 1.0
**Date**: 2025-09-15
**Status**: Ready for ethics submission and regulatory consultation
**Next Review**: Post-ethics approval and site activation
**Contact**: [Principal Investigator] for protocol implementation support