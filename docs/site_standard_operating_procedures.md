# Site Standard Operating Procedures (SOPs)
**Multi-Site EEG Biomarker Validation Study**

---

## Document Information

**Protocol**: Phase IV Multi-Site EEG Biomarker Validation
**SOP Version**: 1.0
**Effective Date**: 2025-09-15
**Review Date**: 2026-09-15
**Scope**: All participating clinical sites

**Participating Sites**:
- University of Leicester (Lead Site)
- MRC BNDU Oxford
- UC San Diego
- Additional sites as approved

---

## 1. EEG Acquisition Standard Operating Procedure

### 1.1 Equipment Requirements

#### Hardware Specifications
**EEG System**: Brain Products actiCHamp or equivalent 32-channel system
**Sampling Rate**: 1000 Hz acquisition, decimated to 250 Hz for analysis
**Electrode Type**: Active Ag/AgCl electrodes
**Cap**: Elastic cap with 10-20 international system layout
**Amplifier**: 24-bit resolution, ±3.125 mV input range

#### Software Requirements
**Acquisition Software**: Brain Vision Recorder v1.21 or higher
**Real-time Processing**: Custom MATLAB/Python pipeline
**Data Storage**: Encrypted local storage with automated backup

### 1.2 Pre-Session Setup

#### Patient Preparation Checklist
- [ ] **Medical History Review**: Confirm inclusion/exclusion criteria
- [ ] **Medication Status**: Document current medications and timing
- [ ] **Informed Consent**: Verify signed consent for EEG recording
- [ ] **UPDRS-III Assessment**: Complete motor examination
- [ ] **Comfort Check**: Explain procedure and address concerns

#### Equipment Preparation
- [ ] **System Calibration**: Daily impedance and signal quality check
- [ ] **Electrode Preparation**: Clean electrodes with alcohol wipes
- [ ] **Cap Sizing**: Select appropriate cap size for patient head circumference
- [ ] **Gel Preparation**: Prepare conductive gel at room temperature
- [ ] **Documentation Ready**: Patient ID, session forms, and data sheets

### 1.3 Electrode Placement and Impedance

#### Standard 10-20 Electrode Positions
**Required Electrodes** (32 positions):
```
Frontal:    Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6
Central:    T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6
Parietal:   P7, P3, Pz, P4, P8, PO9, O1, Oz, O2, PO10
Reference:  Linked mastoids (TP9, TP10)
Ground:     AFz
EOG:        Above/below left eye (vertical), lateral canthi (horizontal)
```

#### Impedance Requirements
**Target Impedance**: <5 kΩ for all electrodes
**Acceptable Range**: <10 kΩ (document if 5-10 kΩ)
**Unacceptable**: >10 kΩ (must reduce before proceeding)

**Impedance Reduction Procedure**:
1. Remove electrode and clean skin with alcohol pad
2. Gently abrade skin with pumice gel or electrode paste
3. Apply fresh conductive gel
4. Reposition electrode and check impedance
5. Repeat until <10 kΩ achieved

### 1.4 Signal Quality Control

#### Real-Time Monitoring
**Visual Inspection**: Continuous monitoring during acquisition
**Artifact Detection**: Automated detection with manual verification
**Signal Range**: ±100 μV typical range (flag if exceeded)
**Frequency Content**: Monitor for 50/60 Hz interference

**Quality Control Metrics**:
- **Electrode Impedance**: Monitored every 5 minutes
- **Signal Amplitude**: Flag excessive movement artifacts >150 μV
- **Frequency Analysis**: Real-time power spectral density check
- **Correlation Check**: Inter-electrode correlation assessment

---

## 2. Common Average Reference (CAR) Procedure

### 2.1 CAR Implementation

#### Mathematical Definition
```
CAR_i(t) = EEG_i(t) - (1/N) × Σ[j=1 to N] EEG_j(t)

Where:
- CAR_i(t) = Common average referenced signal for electrode i
- EEG_i(t) = Raw signal from electrode i
- N = Total number of electrodes (32)
- Σ EEG_j(t) = Sum of all electrode signals
```

#### Implementation Steps
1. **Acquire Raw EEG**: 32-channel raw data at 1000 Hz
2. **Calculate Average**: Compute mean across all channels at each timepoint
3. **Subtract Average**: Remove common average from each channel
4. **Quality Check**: Verify sum of CAR signals ≈ 0

### 2.2 CAR Quality Control

#### Validation Checks
**Sum Verification**: Σ CAR_i(t) should equal zero (within numerical precision)
**Rank Reduction**: CAR reduces data rank from 32 to 31 dimensions
**Reference Independence**: Result should be independent of original reference choice

**Troubleshooting**:
- **Non-zero sum**: Check for bad channels or computation errors
- **Excessive noise**: Verify electrode impedances and connections
- **Channel dropout**: Exclude bad channels before CAR computation

---

## 3. Beta-Burst Threshold Calibration

### 3.1 Threshold Calculation Method

#### Median + 2×MAD Method
```
Threshold = Median(β_envelope) + 2 × MAD(β_envelope)

Where:
- β_envelope = 13-30 Hz Hilbert envelope amplitude
- Median = 50th percentile of envelope values
- MAD = Median Absolute Deviation = Median(|β_envelope - Median(β_envelope)|)
- Factor 2 = Empirically optimized for β-burst detection
```

#### Implementation Procedure
1. **Band-pass Filter**: Apply 13-30 Hz Butterworth filter (order 4)
2. **Hilbert Transform**: Compute analytical signal and amplitude envelope
3. **Baseline Calculation**: Use first 2 minutes of resting data
4. **Statistical Computation**: Calculate median and MAD for each electrode
5. **Threshold Setting**: Set detection threshold = Median + 2×MAD

### 3.2 Threshold Validation

#### Quality Control Checks
**Physiological Range**: Threshold should be 2-10 μV for typical EEG
**Site Consistency**: Thresholds should be comparable across sites (within 2x factor)
**Patient Variability**: Document individual patient threshold values

**Calibration Frequency**:
- **Per Session**: Calculate fresh thresholds for each recording
- **Per Condition**: Separate thresholds for REAL vs SHAM if needed
- **Quality Review**: Weekly review of threshold distributions

### 3.3 Beta-Burst Detection

#### Burst Identification Criteria
**Amplitude Criterion**: β-envelope > threshold
**Duration Criterion**: Minimum duration = 100ms (25 samples at 250 Hz)
**Gap Criterion**: Minimum inter-burst interval = 50ms (12.5 samples)

#### Burst Validation
**Manual Verification**: Visual inspection of detected bursts (10% random sample)
**Statistical Validation**: Burst rate should be 1-10 bursts/second (physiological range)
**Cross-Site Comparison**: Burst detection rates compared across sites

---

## 4. Data Collection Procedures

### 4.1 Session Protocol

#### Baseline Recording (5 minutes)
**Purpose**: Threshold calibration and baseline biomarker assessment
**Conditions**: Eyes open, relaxed sitting, minimal movement
**Instructions**: "Please sit comfortably and look at the center of the screen"

#### REAL Feedback Condition (15 minutes)
**Feedback Type**: Visual feedback directly linked to patient's β-burst activity
**Task**: Simple reaching/pointing with visual targets
**Feedback Latency**: <500ms from burst detection to visual response
**Randomization**: Block randomized order (REAL first vs SHAM first)

#### SHAM Feedback Condition (15 minutes)
**Feedback Type**: Identical visual feedback but randomly generated
**Task**: Same reaching/pointing task as REAL condition
**Patient Blinding**: Patients unaware which condition is active
**Technical Blinding**: Research staff aware for data recording purposes

#### Rest Periods
**Between Conditions**: 5-minute mandatory rest
**Within Conditions**: 2-minute rest every 5 minutes if patient requests
**Movement Check**: Verify electrode impedances after rest periods

### 4.2 Data Recording Standards

#### File Naming Convention
```
Site_PatientID_SessionDate_Condition_Time.eeg
Examples:
LEICESTER_P001_20250315_BASELINE_0900.eeg
LEICESTER_P001_20250315_REAL_0910.eeg
LEICESTER_P001_20250315_SHAM_0930.eeg
```

#### Metadata Documentation
**Required Fields**:
- Patient ID (de-identified)
- Site identifier
- Session date and time
- Condition type (BASELINE/REAL/SHAM)
- Electrode impedances (pre/post session)
- Technical issues or deviations
- UPDRS-III score and medication timing

---

## 5. Quality Control Procedures

### 5.1 Real-Time Quality Control

#### Continuous Monitoring Checklist
- [ ] **Impedance Monitoring**: Check every 5 minutes, flag if >10 kΩ
- [ ] **Artifact Detection**: Monitor for movement, muscle, and eye artifacts
- [ ] **Signal Range**: Verify signals within ±100 μV normal range
- [ ] **Frequency Content**: Check for power line interference (50/60 Hz)
- [ ] **Beta Activity**: Confirm presence of β-band activity (13-30 Hz)

#### Intervention Protocols
**High Impedance**: Stop recording, reduce impedance, resume
**Excessive Artifacts**: Brief pause, patient repositioning, electrode check
**Technical Failure**: Document issue, switch to backup system if available
**Patient Discomfort**: Adjust electrode positions, take brief rest

### 5.2 Post-Session Quality Control

#### Data Validation Checklist
- [ ] **File Integrity**: Verify complete data files and backup copies
- [ ] **Signal Quality**: Review signal traces for major artifacts
- [ ] **Threshold Validation**: Check threshold calculations and detection rates
- [ ] **Burst Detection**: Visual inspection of sample burst detections
- [ ] **Metadata Complete**: Ensure all required documentation completed

#### Quality Metrics
**Acceptable Session Criteria**:
- ≥80% of epochs free from major artifacts
- Valid threshold calculation for ≥90% of electrodes
- Complete data for both REAL and SHAM conditions
- All metadata fields documented

**Session Exclusion Criteria**:
- <50% valid epochs due to artifacts
- Technical failure preventing proper data acquisition
- Major protocol deviations affecting data integrity

---

## 6. Site-Specific Procedures

### 6.1 University of Leicester (Lead Site)

#### Additional Responsibilities
**Central Monitoring**: Real-time quality monitoring for all sites
**Technical Support**: Troubleshooting support for remote sites
**Data Management**: Central database management and backup
**Training**: Initial training and ongoing support for site staff

#### Equipment Configuration
**Primary System**: Brain Products actiCHamp 32-channel
**Backup System**: Secondary identical system available
**Processing Hardware**: Dedicated real-time analysis workstation
**Network**: Secure VPN connection to central database

### 6.2 MRC BNDU Oxford

#### Site-Specific Setup
**Equipment**: Site-specific Brain Products system configuration
**Calibration**: Weekly cross-calibration with Leicester standards
**Data Transfer**: Encrypted daily transfer to central database
**Quality Review**: Weekly quality reports to central monitoring

### 6.3 UC San Diego

#### International Coordination
**Time Zone Management**: Coordinate data collection with UK sites
**Regulatory Compliance**: US IRB requirements and HIPAA compliance
**Equipment Standards**: Identical technical specifications maintained
**Data Transfer**: Secure international data transfer protocols

---

## 7. Training and Certification

### 7.1 Required Training

#### Technical Staff Training (8 hours)
**Module 1**: EEG acquisition principles and equipment operation
**Module 2**: Electrode placement and impedance management
**Module 3**: Signal quality assessment and artifact recognition
**Module 4**: Protocol adherence and data documentation
**Certification**: Written exam (80% pass rate) + practical demonstration

#### Clinical Staff Training (4 hours)
**Module 1**: Patient interaction and consent procedures
**Module 2**: UPDRS-III administration and scoring
**Module 3**: Safety procedures and emergency protocols
**Module 4**: Data privacy and confidentiality requirements

### 7.2 Ongoing Training

#### Monthly Quality Reviews
**Case Studies**: Review challenging data collection cases
**Protocol Updates**: Communication of any protocol amendments
**Performance Feedback**: Individual and site-level performance metrics
**Best Practices**: Sharing of successful techniques across sites

#### Annual Recertification
**Technical Skills**: Practical demonstration of electrode placement
**Protocol Knowledge**: Updated written examination
**Quality Standards**: Review of personal and site quality metrics

---

## 8. Data Management and Documentation

### 8.1 Case Report Forms (CRF)

#### Required Documentation
**Demographic Form**: Age, sex, disease duration, medications
**Session Forms**: Date, time, conditions, technical notes
**Quality Control Forms**: Impedances, signal quality, artifacts
**Adverse Events**: Any issues or complications during sessions
**Protocol Deviations**: Documentation and impact assessment

### 8.2 Data Security

#### Local Data Storage
**Encryption**: AES-256 encryption for all stored data
**Access Control**: Role-based access with individual login credentials
**Backup**: Daily automated backup to secure local storage
**Retention**: Local data retained until central transfer confirmed

#### Data Transfer
**Frequency**: Daily transfer to central database
**Method**: Secure FTP with end-to-end encryption
**Verification**: Checksum verification of transferred files
**Cleanup**: Local data securely deleted after successful transfer

---

## 9. Troubleshooting Guide

### 9.1 Common Technical Issues

#### Poor Signal Quality
**Symptoms**: Excessive noise, artifacts, or signal drift
**Causes**: High impedance, poor electrode contact, movement
**Solutions**:
1. Check and reduce electrode impedances
2. Reposition electrodes with fresh gel
3. Ensure patient comfort and minimal movement
4. Check for electrical interference sources

#### Beta-Burst Detection Issues
**Symptoms**: Unusually high/low burst detection rates
**Causes**: Threshold calculation errors, filter issues, medication effects
**Solutions**:
1. Recalculate thresholds from baseline data
2. Verify 13-30 Hz filter is correctly applied
3. Check patient medication status
4. Visual verification of detected bursts

### 9.2 Emergency Procedures

#### Patient Distress
**Immediate Actions**:
1. Stop recording immediately
2. Remove EEG cap if requested
3. Provide patient support and reassurance
4. Document incident in adverse event log
5. Contact principal investigator within 24 hours

#### Equipment Failure
**Backup Procedures**:
1. Switch to backup EEG system if available
2. Document failure and troubleshooting attempts
3. Contact technical support immediately
4. Reschedule patient if backup unavailable
5. Report to central monitoring within 4 hours

---

## 10. Contact Information

### 10.1 Technical Support
**Primary Contact**: [Technical Director Name]
**Phone**: [24/7 technical support number]
**Email**: [technical-support@study-email.com]
**Response Time**: <2 hours during business hours, <4 hours off-hours

### 10.2 Clinical Support
**Principal Investigator**: [PI Name]
**Phone**: [PI contact number]
**Email**: [PI email address]
**Medical Monitor**: [Medical Monitor Name]
**Emergency Contact**: [24/7 clinical support]

---

## 11. Version Control and Updates

### 11.1 Document Revision History
| Version | Date | Changes | Approved By |
|---------|------|---------|-------------|
| 1.0 | 2025-09-15 | Initial version | [PI Name] |
| | | | |

### 11.2 Update Procedures
**Amendment Process**: Protocol amendments communicated within 48 hours
**Training Updates**: Mandatory retraining for major procedural changes
**Site Notification**: All sites notified simultaneously of updates
**Version Control**: Previous versions archived with change documentation

---

**SOP Status**: Ready for implementation
**Next Review**: 6 months post-study initiation
**Approval Required**: Site PI signature before study initiation