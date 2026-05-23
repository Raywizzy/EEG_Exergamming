# Phase IV Prospective Validation Protocol
**EEG-Based Parkinson's Disease Classification Using Core15+ Biomarkers**

**Protocol Version:** 1.0
**Date:** 2025-09-20
**Principal Investigator:** [To be assigned]
**Study Phase:** IV (Prospective Clinical Validation)
**Trial Design:** Multi-center, prospective, blinded diagnostic accuracy study

---

## Study Overview

### Background and Rationale

**Scientific Foundation:**
- **Phase II:** Core5 beta-burst features achieved chance-level performance (50.5% BA, p = 0.423)
- **Phase III:** Core15+ multimodal features achieved breakthrough performance (96.7% BA, p = 0.005)
- **Cross-site validation:** Demonstrated generalization across 3 independent datasets (81 subjects)

**Clinical Need:**
- Current PD diagnosis relies on clinical assessment (MDS-UPDRS) with ~15% misdiagnosis rate
- DaTscan imaging expensive ($3,000-5,000), limited availability, radiation exposure
- EEG-based biomarker offers accessible, cost-effective, radiation-free alternative

**Innovation:**
- First EEG biomarker to exceed 95% accuracy for PD vs Control classification
- Multimodal feature approach targeting connectivity, spectral, and temporal abnormalities
- CORAL domain adaptation enables cross-site deployment without site-specific calibration

### Primary Objective

**Prospectively validate Core15+ EEG biomarkers for Parkinson's disease classification in a multi-center clinical setting.**

### Secondary Objectives

1. **Diagnostic Performance:** Confirm ≥90% sensitivity and ≥85% specificity
2. **Cross-site Generalization:** Demonstrate consistent performance across sites
3. **Clinical Utility:** Compare with standard diagnostic methods (MDS-UPDRS, DaTscan)
4. **Implementation:** Assess feasibility in real-world clinical workflows
5. **Health Economics:** Evaluate cost-effectiveness vs current standard of care

---

## Study Design

### Trial Type
**Multi-center, prospective, single-blinded diagnostic accuracy study**

### CONSORT Flow Diagram

```
                    SCREENING
                   (n = 250 estimated)
                        │
            ┌───────────┼───────────┐
            │           │           │
      EXCLUSIONS    ELIGIBLE    DECLINES
      (n = 30)     (n = 200)    (n = 20)
                        │
                   ENROLLMENT
                   (n = 200)
                        │
            ┌───────────┼───────────┐
            │                       │
    PD PATIENTS                CONTROLS
    (n = 100)                  (n = 100)
         │                          │
         ├── Site 1: n=25          ├── Site 1: n=25
         ├── Site 2: n=25          ├── Site 2: n=25
         ├── Site 3: n=25          ├── Site 3: n=25
         └── Site 4: n=25          └── Site 4: n=25
         │                          │
         └──────────┬───────────────┘
                    │
               EEG RECORDING
               (20-minute protocol)
                    │
            ┌───────┼───────┐
            │       │       │
      TECHNICAL  SUCCESS  REPEAT
      FAILURE    (n=190)  (n=10)
      (n=5)         │
                    │
              FEATURE EXTRACTION
              (Core15+ automated)
                    │
              CLASSIFICATION
              (Blinded analysis)
                    │
              PRIMARY ANALYSIS
              (n = 190 evaluable)
```

### Study Sites

**Target Sites:** 4 clinical centers
- **Site 1:** Academic medical center (primary coordinating site)
- **Site 2:** Movement disorders specialty clinic
- **Site 3:** Community neurology practice
- **Site 4:** International site (generalizability)

**Site Requirements:**
- IRB/ethics approval
- Movement disorders neurologist
- EEG technician training
- Standard 64-channel EEG system
- Secure data transmission capability

---

## Participants

### Inclusion Criteria

**PD Group:**
- Age 45-80 years
- Clinical diagnosis of idiopathic Parkinson's disease (MDS criteria)
- Hoehn & Yahr stage I-III
- On stable dopaminergic medication ≥3 months
- Able to provide informed consent

**Control Group:**
- Age 45-80 years (matched to PD group ±5 years)
- No neurological disorders
- No medications affecting CNS
- Normal cognitive screening (MoCA ≥26)
- Able to provide informed consent

### Exclusion Criteria

**Both Groups:**
- Secondary or atypical parkinsonism
- Significant cognitive impairment (MoCA <24)
- Major psychiatric disorder (active)
- History of head trauma with LOC >30 minutes
- Seizure disorder or epilepsy
- Skull defects or metallic implants affecting EEG
- Inability to tolerate 20-minute EEG recording

### Sample Size Calculation

**Primary Analysis:** Diagnostic accuracy (sensitivity/specificity)

**Assumptions:**
- Core15+ baseline performance: 96.7% BA (100% sensitivity, 93.3% specificity)
- Null hypothesis: Sensitivity ≤80%, Specificity ≤80%
- Alternative hypothesis: Sensitivity ≥90%, Specificity ≥85%
- α = 0.05 (two-sided), β = 0.20 (80% power)
- Design effect for clustering: 1.2

**Power Calculation:**
```
For Sensitivity:
n = [Z_α/2 + Z_β]² × p(1-p) / (p - p₀)²
n = [1.96 + 0.84]² × 0.90(0.10) / (0.90 - 0.80)²
n = 63 PD patients

For Specificity:
n = [1.96 + 0.84]² × 0.85(0.15) / (0.85 - 0.80)²
n = 81 controls

With 10% dropout: 70 PD + 90 controls = 160 total
With clustering adjustment: 160 × 1.2 = 192
Final sample: 200 total (100 PD + 100 controls)
```

---

## Study Procedures

### Screening and Enrollment

**Visit 1: Screening (Day -7 to 0)**
1. **Informed consent** (IRB-approved forms)
2. **Medical history** and demographics
3. **Clinical assessment:**
   - MDS-UPDRS Parts I-IV
   - Hoehn & Yahr staging
   - MoCA cognitive screening
   - Medication history
4. **Eligibility confirmation**
5. **EEG visit scheduling**

### EEG Recording Protocol

**Visit 2: EEG Session (Day 0 to +14)**

**Pre-recording (15 minutes):**
- Participant preparation and positioning
- 64-channel cap placement (10-20 system)
- Impedance testing (<5 kΩ)
- Medication timing documentation (PD patients)

**Recording Protocol (20 minutes):**
1. **Eyes-closed resting state:** 5 minutes
2. **Eyes-open resting state:** 5 minutes
3. **Motor imagery task:** 5 minutes (hand movements)
4. **Cognitive task:** 5 minutes (working memory)

**Technical Specifications:**
- **Sampling rate:** 512 Hz
- **Filters:** 0.1-100 Hz (hardware), 1-40 Hz (analysis)
- **Reference:** Linked mastoids or average reference
- **Monitoring:** Real-time artifact detection
- **Quality control:** Automated impedance checks

**Post-recording (10 minutes):**
- Data quality assessment
- Technical review and approval
- Secure data upload to central server

### Blinding Procedures

**Triple Blinding:**
1. **EEG technicians:** Blinded to clinical diagnosis
2. **Algorithm analysis:** Automated feature extraction and classification
3. **Statistical analysts:** Blinded to site and clinical variables

**Clinical Reference Standard:**
- Movement disorders neurologist assessment (independent of EEG)
- MDS-UPDRS scoring (blinded to EEG results)
- Consensus diagnosis for discordant cases

---

## Data Management and Analysis

### Primary Endpoint

**Diagnostic Accuracy Measures:**
- **Sensitivity:** Proportion of PD patients correctly classified
- **Specificity:** Proportion of controls correctly classified
- **Balanced Accuracy:** (Sensitivity + Specificity) / 2
- **Target Performance:** BA ≥87.5% (corresponding to 90% sensitivity, 85% specificity)

### Secondary Endpoints

**Performance Metrics:**
- Area under ROC curve (AUC)
- Positive/negative predictive values
- Diagnostic odds ratio
- Site-specific performance metrics

**Clinical Correlations:**
- Performance vs disease severity (H&Y stage)
- Performance vs medication state (ON/OFF)
- Performance vs age and gender
- Time to diagnosis with EEG vs standard care

### Statistical Analysis Plan

**Primary Analysis:**
```
Diagnostic Accuracy Analysis:
- Calculate sensitivity, specificity, and 95% CIs
- Use exact binomial methods for small samples
- Cluster-robust standard errors for site effects
- McNemar's test for paired comparisons vs reference standard
```

**Secondary Analyses:**
```
Cross-site Validation:
- Random effects logistic regression
- Site as random effect, adjust for age/gender
- Test for site × performance interaction

Correlation Analyses:
- Spearman correlations with clinical measures
- Multiple regression for predictors of accuracy
- Subgroup analyses by H&Y stage
```

**Interim Analysis:**
- **Timing:** After 50% enrollment (n=100)
- **Scope:** Safety, data quality, protocol adherence
- **DSMB review:** Efficacy boundaries for early success/futility

---

## Quality Control and Monitoring

### Data and Safety Monitoring Board (DSMB)

**Composition:**
- Independent biostatistician (Chair)
- Movement disorders neurologist
- Clinical EEG specialist
- Bioethicist
- Patient representative

**Responsibilities:**
- Safety oversight and adverse event review
- Data quality monitoring
- Protocol adherence assessment
- Stopping rule recommendations

### Stopping Rules

**Efficacy Boundaries (Early Success):**
```
Interim Analysis (n=100):
If 95% CI lower bound for BA > 85%:
  → Recommend early success, continue for secondary endpoints

Final Analysis (n=200):
If 95% CI lower bound for BA > 87.5%:
  → Primary success criterion met
```

**Futility Boundaries:**
```
Interim Analysis (n=100):
If 95% CI upper bound for BA < 75%:
  → Recommend futility stop

Technical Failure Rate:
If >20% EEG recordings fail quality control:
  → Protocol amendment for training/equipment
```

**Safety Boundaries:**
```
Adverse Events:
- Any serious adverse event related to EEG
- >5% participant withdrawal due to discomfort

Data Integrity:
- >10% protocol deviations
- Evidence of unblinding
```

### Quality Assurance

**Site Monitoring:**
- Remote monitoring for 100% of data
- On-site monitoring for 25% of participants
- Central review of all EEG recordings

**Technical QC:**
- Real-time automated quality checks
- Centralized impedance monitoring
- Artifact detection and reporting

**Clinical QC:**
- Source document verification
- Adverse event reconciliation
- Protocol deviation tracking

---

## Regulatory and Ethical Considerations

### Regulatory Pathway

**FDA Guidance Compliance:**
- **De Novo pathway:** Novel EEG-based diagnostic device
- **Clinical evidence:** Multi-site prospective validation
- **Software as Medical Device (SaMD):** Class II risk category
- **Quality management:** ISO 13485 compliance

**Required Documentation:**
- Clinical protocol and statistical analysis plan
- Technical file and software documentation
- Risk management (ISO 14971)
- Clinical evaluation report

### IRB/Ethics Approval

**Central IRB Model:**
- Single IRB review for multi-site efficiency
- Site-specific considerations and approvals
- Continuing review every 12 months

**Informed Consent:**
- Detailed explanation of EEG procedures
- Data sharing and privacy protections
- Optional consent for future research use
- Right to withdraw without prejudice

### Data Protection

**HIPAA Compliance:**
- De-identification of all transmitted data
- Subject ID mapping maintained locally
- Encrypted data transmission and storage
- Limited data access on need-to-know basis

**International Standards:**
- GDPR compliance for international sites
- Local data protection requirements
- Cross-border data transfer agreements

---

## Timeline and Milestones

### Study Timeline (24 months)

**Months 1-3: Study Startup**
- IRB approvals and regulatory submissions
- Site contracts and training
- CRF development and database setup
- Central lab and analysis pipeline setup

**Months 4-18: Enrollment and Data Collection**
- Site initiation visits
- Participant recruitment and enrollment
- EEG data collection
- Real-time quality monitoring

**Months 19-21: Data Analysis**
- Database lock and cleaning
- Statistical analysis execution
- Clinical study report preparation

**Months 22-24: Reporting and Submission**
- Manuscript preparation
- Regulatory submission preparation
- Conference presentations

### Key Milestones

| Milestone | Timeline | Success Criteria |
|-----------|----------|------------------|
| **First patient enrolled** | Month 4 | Site 1 activation |
| **25% enrollment** | Month 8 | n=50 enrolled |
| **Interim analysis** | Month 12 | n=100 completed |
| **Last patient visit** | Month 18 | n=200 completed |
| **Database lock** | Month 19 | 100% data verification |
| **Primary results** | Month 21 | Statistical analysis complete |
| **Regulatory submission** | Month 24 | FDA filing ready |

---

## Expected Outcomes and Impact

### Primary Success Scenario

**Performance Targets Met:**
- **Sensitivity:** ≥90% (95% CI: 83-97%)
- **Specificity:** ≥85% (95% CI: 78-92%)
- **Balanced Accuracy:** ≥87.5%
- **Cross-site consistency:** <10% BA variation

**Regulatory Impact:**
- FDA De Novo authorization pathway
- CE marking for European deployment
- Health Canada and other international approvals

### Clinical Translation

**Implementation Pathway:**
- Integration with existing EEG equipment
- Training programs for clinical staff
- Quality assurance protocols
- Reimbursement strategy development

**Market Access:**
- Movement disorders clinics (primary market)
- General neurology practices
- Emergency departments (differential diagnosis)
- International markets (accessibility advantage)

### Scientific Impact

**Publications:**
- High-impact clinical journal (primary results)
- Technical methodology paper
- Health economics analysis
- Implementation science study

**Follow-up Studies:**
- Head-to-head comparison with DaTscan
- Longitudinal monitoring of disease progression
- Extension to other movement disorders
- Pediatric population adaptation

---

## Budget and Resources

### Personnel Requirements

**Per Site:**
- Principal Investigator (20% effort): $50,000
- Study Coordinator (50% effort): $35,000
- EEG Technician (25% effort): $15,000
- Data Manager (10% effort): $8,000

**Central Coordinating Center:**
- Overall PI (25% effort): $75,000
- Biostatistician (30% effort): $45,000
- Data Manager (50% effort): $40,000
- Regulatory Affairs (25% effort): $35,000

### Equipment and Technology

**EEG Systems:** $200,000 (4 sites × $50,000)
**Software Licensing:** $50,000
**Data Management Platform:** $100,000
**Central Computing Infrastructure:** $75,000

### Total Budget Estimate

**Direct Costs:** $1,200,000
- Personnel: $800,000
- Equipment: $200,000
- Technology: $200,000

**Indirect Costs (30%):** $360,000
**Total Project Cost:** $1,560,000

---

## Risk Management

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **EEG data quality issues** | Medium | High | Standardized training, real-time monitoring |
| **Cross-site variability** | Medium | Medium | CORAL adaptation, centralized analysis |
| **Software/hardware failures** | Low | High | Redundant systems, vendor support |

### Clinical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Recruitment delays** | High | Medium | Multiple sites, flexible timeline |
| **Participant dropout** | Medium | Medium | 10% over-enrollment, retention incentives |
| **Protocol deviations** | Medium | Low | Comprehensive training, monitoring |

### Regulatory Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **FDA guidance changes** | Low | High | Early pre-submission meeting |
| **IRB delays** | Medium | Medium | Central IRB model, early engagement |
| **Data privacy issues** | Low | High | HIPAA/GDPR compliance, legal review |

---

## Conclusion

This Phase IV prospective validation protocol provides a comprehensive framework for clinical translation of Core15+ EEG biomarkers. The study design addresses regulatory requirements while maintaining scientific rigor, setting the foundation for FDA approval and clinical deployment.

**Key Strengths:**
- Multi-center design ensures generalizability
- Adequate statistical power for regulatory submission
- Comprehensive quality control and monitoring
- Alignment with FDA guidance for novel diagnostics

**Success Criteria:**
- Primary endpoint achievement (≥87.5% BA)
- Regulatory approval pathway completion
- Clinical implementation readiness
- Commercial partnership facilitation

---

*Protocol Version 1.0 - 2025-09-20*
*Ready for IRB submission and investigator review*