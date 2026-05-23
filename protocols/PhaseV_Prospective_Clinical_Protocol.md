# Phase V Prospective Clinical Protocol

**Title:** Prospective multi-site validation of a locked EEG biomarker pipeline (Core5 + CORAL) for Parkinson's disease classification

**Protocol Version:** 1.0
**Date:** 2025-09-19
**Principal Investigator:** [PI Name]
**Co-Principal Investigators:** [Co-PI Names]
**Sponsor:** [Institution]
**Registration:** ClinicalTrials.gov [TBD] / ISRCTN [TBD]

---

## 1. OBJECTIVES & ENDPOINTS

### Primary Objective
Validate prospective, real-world performance of the locked Core5 + CORAL pipeline for Parkinson's disease classification across independent clinical sites.

### Primary Endpoint
**Balanced Accuracy (BA)** on prospectively acquired data using Leave-One-Site-Out cross-validation, with prespecified clinical target ≥65% and hypothesis test vs 50% chance performance.

### Key Secondary Endpoints
- **Inference latency** <500 ms per subject (pipeline runtime requirement)
- **Class-wise performance:** Sensitivity, specificity, ROC-AUC (descriptive)
- **Site stability:** Weekly monitoring with alert if any site BA drops >10pp below mean across two consecutive weeks
- **Data quality metrics:** Proportion of subjects passing automated QC pipeline
- **Safety assessment:** Non-interventional device/software monitoring

---

## 2. STUDY DESIGN & SETTING

**Design:** Prospective, multi-centre, diagnostic accuracy validation study
**Setting:** 3-5 independent clinical research sites
**Blinding:** Analysts blinded to clinical labels during feature extraction; labels revealed only for final analysis
**Pipeline Status:** Locked - no model retraining or parameter optimization after trial initiation

**Planned Sites:**
- [Site 1: Institution, Country]
- [Site 2: Institution, Country]
- [Site 3: Institution, Country]
- [Optional Site 4: Institution, Country]
- [Optional Site 5: Institution, Country]

---

## 3. STUDY POPULATION

### Inclusion Criteria
- **Parkinson's Disease:** Idiopathic PD meeting MDS clinical criteria
- **Healthy Controls:** Age-matched controls without neurological conditions
- **Age:** 40-85 years
- **Capacity:** Able to provide written informed consent
- **Compliance:** Able to sit quietly for resting-state EEG recording

### Exclusion Criteria
- Epilepsy or major neurological comorbidity
- Active implanted neurostimulators during EEG recording
- Uncorrected severe vision/hearing impairment affecting compliance
- Inability to remain still for minimum 3-minute recording period
- Current participation in interventional clinical trials

---

## 4. PROCEDURES

### EEG Acquisition Protocol
- **Recording Type:** Resting-state, eyes-open
- **System:** 64-channel (or site-standard with ≥19 channels)
- **Sampling Rate:** ≥500 Hz (downsampled to 256 Hz for analysis)
- **Reference:** Common Average Reference (CAR)
- **Duration:** 2-3 minutes (minimum 120s artifact-free target)
- **Environment:** Electrically shielded, quiet room

### Data Processing Pipeline (Locked)
1. **BIDS Import:** Standardized data structure
2. **Preprocessing:** 1-45 Hz bandpass, 60 Hz notch, CAR reference
3. **QC Assessment:** Automated artifact detection and rejection
4. **Core5 Extraction:** Locked beta-burst feature computation
5. **CORAL Adaptation:** Domain harmonization (training site → test site)
6. **Classification:** Locked logistic regression with balanced weights

### Quality Control
- **Real-time monitoring:** Automated QC dashboard
- **Weekly reports:** Site-specific performance tracking
- **Alert system:** Automatic notification if QC thresholds breached

---

## 5. SAMPLE SIZE & STATISTICAL POWER

### Target Sample Size
**Total Target:** 120-200 subjects across all sites
**Composition:** Balanced PD and HC groups where feasible
**Per-Site Minimum:** 20-40 subjects (site-dependent)

### Power Calculation
**Primary Test:** One-sided t-test of LOSO fold BAs vs 50% chance
**Design Parameters:**
- Target effect: μ = 65% BA
- Null hypothesis: μ ≤ 50%
- Alpha level: 0.05 (one-sided)
- Expected SD: 8-10 percentage points (from Phase IV data)

| Sites (k) | SD (pp) | Mean BA Target | Power (α=0.05) |
|-----------|---------|---------------|-----------------|
| 3         | 8%      | 65%           | ~85%            |
| 4         | 8%      | 65%           | ~90%            |
| 3         | 10%     | 65%           | ~80%            |
| 4         | 10%     | 65%           | ~85%            |

### Planned Recruitment by Site

| Site | Target N (PD) | Target N (HC) | Total |
|------|---------------|---------------|-------|
| [Site 1] | [ ] | [ ] | [ ] |
| [Site 2] | [ ] | [ ] | [ ] |
| [Site 3] | [ ] | [ ] | [ ] |
| [Site 4] | [ ] | [ ] | [ ] |
| **Total** | **[ ]** | **[ ]** | **[ ]** |

---

## 6. STATISTICAL ANALYSIS PLAN

### Primary Analysis
**Hypothesis Test:** One-sided one-sample t-test across LOSO folds
- H₀: μ_BA ≤ 50% (chance performance)
- H₁: μ_BA > 50% (above chance)
- Significance level: α = 0.05

**Effect Size:** Cohen's d relative to 50% chance performance
**Confidence Intervals:** 95% CI for mean BA using t-distribution

### Secondary Analyses
- **Performance Metrics:** Sensitivity, specificity, positive/negative predictive values
- **ROC Analysis:** Area under curve with 95% CI
- **Site Consistency:** Descriptive statistics per site, range assessment
- **Processing Metrics:** Inference latency statistics, QC pass rates

### Missing Data Strategy
- **QC Failures:** Pre-defined exclusion rules applied automatically
- **Technical Issues:** CONSORT flow diagram documenting all exclusions
- **Analysis Population:** Modified intention-to-treat (all subjects with valid EEG data)

---

## 7. DATA GOVERNANCE & MONITORING

### Data Security
- **Encryption:** AES-256 at rest and in transit
- **Access Control:** VPN-secured, role-based permissions
- **Audit Trail:** Complete logging of all data access and processing
- **GDPR Compliance:** Full adherence to data protection regulations

### Quality Monitoring
- **Independent DSMB:** Data and Safety Monitoring Board oversight
- **Meeting Schedule:** Study initiation, mid-study review, study completion
- **Interim Monitoring:** Monthly safety and quality reports
- **Automated QC:** Real-time dashboard with alert system

### Data Management
- **Anonymization:** De-identified subject IDs only
- **BIDS Standard:** Standardized neuroimaging data structure
- **Checksums:** Data integrity verification at all transfer points
- **Backup Strategy:** Multi-site redundant storage with version control

---

## 8. ETHICS & REGULATORY

### Risk Classification
**Risk Level:** Minimal risk (non-interventional EEG analysis)
**Device Classification:** Non-significant risk analytical software
**Intervention:** None (observational study only)

### Regulatory Approvals
- **IRB/REC:** Local institutional review board approval at each site
- **Sponsor Oversight:** Central ethics coordination and monitoring
- **Protocol Amendments:** Formal approval process for any modifications

### Informed Consent
- **Process:** Written informed consent obtained at each site
- **Documentation:** Signed consent forms maintained per local requirements
- **Withdrawal:** Participants may withdraw at any time without penalty

---

## 9. SUCCESS CRITERIA & GO/NO-GO DECISIONS

### Primary Success Criteria (GO)
- **Performance:** Overall LOSO mean BA ≥65%
- **Statistical Significance:** p < 0.05 vs 50% chance (one-sided test)
- **Technical:** Processing latency <500 ms per subject
- **Consistency:** No site with catastrophically low performance (<55% BA)

### Conditional Success (ITERATE)
- **Performance:** Mean BA 60-65% with consistent site performance
- **Action:** Review and refine acquisition SOPs, consider expanded cohort
- **Timeline:** 3-month protocol amendment period

### Study Termination Criteria (NO-GO)
- **Performance:** Mean BA <60% or high between-site inconsistency
- **Technical:** Persistent QC failures or processing issues
- **Safety:** Any safety concerns identified by DSMB

---

## 10. TIMELINE & MILESTONES

### Pre-Study Phase (Months -2 to 0)
- Ethics approvals and regulatory submissions
- Site training and SOP standardization
- System installation and validation testing

### Recruitment Phase (Months 1-3)
- Active subject recruitment and data collection
- Real-time QC monitoring and weekly reports
- Interim DSMB safety review

### Analysis Phase (Month 4)
- Database lock and final QC verification
- Statistical analysis execution
- DSMB final review and recommendations

### Reporting Phase (Month 5)
- Results presentation to stakeholders
- Manuscript preparation and submission
- Regulatory pre-submission package preparation

---

## 11. ROLES & RESPONSIBILITIES

### Principal Investigator
- Overall study oversight and scientific leadership
- Protocol compliance and quality assurance
- Regulatory and ethics coordination

### Site Principal Investigators
- Local recruitment and consent procedures
- EEG data acquisition per protocol SOP
- Secure data transfer and site QC

### Data Management Team
- BIDS standardization and data integrity
- Automated QC pipeline maintenance
- Audit trail documentation

### Statistical Analysis Team
- Blinded analysis execution
- Results interpretation and reporting
- Manuscript statistical content

### Data Safety Monitoring Board
- Independent safety and efficacy oversight
- Protocol adherence monitoring
- Go/no-go decision recommendations

---

## APPENDICES

### Appendix A: Core5 Feature Definitions
[Mathematical definitions of the five beta-burst biomarkers]

### Appendix B: CORAL Domain Adaptation Algorithm
[Technical specifications of the harmonization method]

### Appendix C: Standard Operating Procedures
[Detailed EEG acquisition and processing protocols]

### Appendix D: Quality Control Metrics
[Automated QC thresholds and alert criteria]

### Appendix E: CONSORT Flow Diagram Template
[Subject inclusion/exclusion tracking framework]

---

**Protocol Status:** DRAFT v1.0
**Next Review:** [Date]
**Approval Required:** Site IRBs, Sponsor QA, DSMB Charter

---

*This protocol integrates seamlessly with the validated Phase IV automation infrastructure, ensuring continuity from retrospective validation to prospective clinical deployment.*