# Monitoring & Data Management Plan
**Phase IV Multi-Site EEG Biomarker Validation Study**

---

## Document Information

**Protocol Title**: Prospective Multi-Site Validation of EEG Beta-Burst Biomarkers for Parkinson's Disease Assessment
**Plan Version**: 1.0
**Effective Date**: 2025-09-15
**Principal Investigator**: [Name]
**Institution**: University of Leicester
**Data Management Team**: [Team Lead Name]

---

## 1. Study Monitoring Overview

### 1.1 Monitoring Strategy
**Monitoring Approach**: Risk-based monitoring with centralized and on-site components
**Risk Level**: Medium risk - non-interventional study with medical device software
**Monitoring Frequency**: Monthly remote monitoring + quarterly on-site visits
**Quality Focus**: Data integrity, protocol compliance, patient safety

### 1.2 Monitoring Objectives
1. **Protocol Compliance**: Ensure adherence to approved study protocol and SOPs
2. **Data Quality**: Verify accuracy, completeness, and consistency of study data
3. **Regulatory Compliance**: Confirm adherence to GCP, local regulations, and ethics requirements
4. **Patient Safety**: Monitor for adverse events and protocol deviations
5. **Site Performance**: Assess recruitment, retention, and operational metrics

### 1.3 Monitoring Team Structure
**Study Monitor**: [Name] - Overall monitoring coordination
**Medical Monitor**: [Name] - Clinical oversight and safety review
**Data Manager**: [Name] - Database management and quality control
**Technical Monitor**: [Name] - EEG system performance and data quality

---

## 2. Data Management System Architecture

### 2.1 Data Management Infrastructure

#### Central Database System
**Platform**: REDCap (Research Electronic Data Capture)
**Version**: REDCap v13.0 or higher
**Hosting**: University of Leicester secure servers
**Backup**: Daily automated backups with 7-day retention
**Access Control**: Role-based permissions with audit logging

#### Data Flow Architecture
```
Site Data Collection → Local Validation → Secure Transfer → Central Database → Quality Control → Analysis
```

**Data Types**:
1. **EEG Raw Data**: Original acquisition files (.eeg, .vhdr, .vmrk)
2. **Processed Data**: Beta envelopes, burst detections, biomarker values
3. **Clinical Data**: Demographics, UPDRS-III, medication logs
4. **Quality Control Data**: Signal quality metrics, protocol deviations
5. **Metadata**: Session logs, technical parameters, staff notes

### 2.2 Data Security Framework

#### Technical Security
**Encryption**: AES-256 encryption for data at rest and in transit
**Network Security**: VPN tunnels for all data transfers
**Access Control**: Multi-factor authentication required
**Audit Logging**: Complete activity logs with user identification
**Backup Security**: Encrypted backups stored in separate secure facility

#### Administrative Security
**Data Access Agreement**: Signed agreements for all personnel
**Role-Based Permissions**: Minimum necessary access principles
**Regular Security Training**: Annual training for all staff
**Incident Response**: Documented procedures for security breaches
**Compliance Auditing**: Quarterly security compliance reviews

---

## 3. Data Collection Procedures

### 3.1 Case Report Form (CRF) Design

#### Electronic CRF Structure
**Patient Screening Form**:
- Demographics, medical history, inclusion/exclusion criteria
- Informed consent documentation
- Baseline UPDRS-III assessment

**Session Forms**:
- Pre-session checklist and patient preparation
- EEG acquisition parameters and quality metrics
- Technical issues and protocol deviations
- Post-session assessment and next steps

**Adverse Event Forms**:
- Event description, severity, causality assessment
- Actions taken and outcome
- Follow-up requirements

#### Data Entry Requirements
**Real-Time Entry**: All CRF data entered within 24 hours of collection
**Source Documentation**: Direct entry from source documents when possible
**Edit Checks**: Automated range and consistency checks implemented
**Missing Data**: Required fields enforced with explanation for missing data

### 3.2 Data Validation Procedures

#### Automated Data Validation
**Range Checks**: Physiological ranges for all numeric values
**Consistency Checks**: Cross-field validation for related data points
**Completeness Checks**: Missing data identification and flagging
**Logic Checks**: Protocol compliance validation (e.g., session order)

#### Manual Data Review
**Source Data Verification (SDV)**:
- 100% SDV for primary endpoint data
- 20% SDV for secondary endpoint data
- 10% SDV for descriptive data
- Risk-based SDV targeting high-risk data points

**Clinical Data Review**:
- Weekly review of all adverse events
- Monthly review of protocol deviations
- Quarterly comprehensive data review
- Real-time review of safety-relevant data

---

## 4. Remote Monitoring Procedures

### 4.1 Centralized Monitoring Activities

#### Real-Time Data Monitoring
**Daily Activities**:
- Review of overnight data transfers
- EEG signal quality assessment
- Protocol deviation identification
- Adverse event monitoring
- Site performance metrics review

**Weekly Activities**:
- Comprehensive data quality review
- Site recruitment and retention analysis
- Cross-site consistency assessment
- Technical performance evaluation
- Communication with site teams

#### Monitoring Reports
**Monthly Site Reports**:
- Enrollment progress and projections
- Data quality metrics and trends
- Protocol compliance assessment
- Technical performance summary
- Action items and recommendations

**Quarterly Study Reports**:
- Overall study progress assessment
- Cross-site performance comparison
- Data quality summary statistics
- Safety and adverse event analysis
- Risk assessment and mitigation strategies

### 4.2 Site Communication Procedures

#### Regular Communication Schedule
**Daily**: Technical support availability (email/phone)
**Weekly**: Site check-in calls with study coordinator
**Monthly**: Formal monitoring review with site PI
**Quarterly**: Comprehensive site performance review
**Ad-hoc**: Immediate response to critical issues or queries

#### Communication Documentation
**Meeting Minutes**: All formal communications documented
**Action Item Tracking**: Systematic follow-up on identified issues
**Training Records**: Documentation of all training activities
**Issue Resolution**: Complete audit trail for problem resolution

---

## 5. On-Site Monitoring

### 5.1 Site Visit Schedule

#### Initial Site Activation Visit
**Timing**: Before first patient enrollment
**Duration**: 2 days
**Objectives**:
- Site readiness assessment
- Staff training and certification
- Equipment validation and calibration
- Review of local procedures and documentation
- IRB/ethics approval verification

#### Routine Monitoring Visits
**Frequency**: Quarterly during active enrollment
**Duration**: 1 day per visit
**Focus Areas**:
- Source data verification
- Protocol compliance assessment
- Staff performance evaluation
- Equipment maintenance verification
- Regulatory file review

#### Close-Out Visit
**Timing**: After completion of data collection
**Duration**: 1 day
**Activities**:
- Final data verification
- Equipment return/transfer
- Regulatory file completion
- Final site assessment and feedback

### 5.2 Site Visit Procedures

#### Pre-Visit Planning
**Site Notification**: 2 weeks advance notice
**Visit Agenda**: Detailed agenda shared with site
**Document Review**: Remote pre-screening of key documents
**Issue Preparation**: Identified issues for on-site resolution
**Resource Planning**: Required materials and personnel

#### During Site Visits
**Opening Meeting**: Review visit objectives and agenda
**Document Review**: Systematic review of regulatory files
**Source Data Verification**: Comparison of CRF data with source
**Staff Interviews**: Discussion with key site personnel
**Equipment Inspection**: Technical validation of EEG systems
**Closing Meeting**: Summary of findings and action items

#### Post-Visit Activities
**Site Visit Report**: Detailed report within 5 business days
**Action Item List**: Specific items requiring site response
**Timeline for Responses**: Clear deadlines for corrective actions
**Follow-Up Communications**: Systematic follow-up until resolution

---

## 6. Data Quality Control

### 6.1 Quality Control Framework

#### Multi-Level QC Approach
**Level 1 - Real-Time QC**: Immediate validation during data collection
**Level 2 - Daily QC**: Automated checks after data transfer
**Level 3 - Weekly QC**: Manual review and trend analysis
**Level 4 - Monthly QC**: Comprehensive quality assessment

#### Quality Metrics
**Data Completeness**: % of required fields completed
**Data Accuracy**: % of values within expected ranges
**Protocol Compliance**: % of procedures following protocol
**Technical Performance**: % of sessions meeting quality standards
**Cross-Site Consistency**: Variability measures across sites

### 6.2 EEG Data Quality Control

#### Signal Quality Assessment
**Automated Metrics**:
- Signal amplitude statistics (mean, std, range)
- Frequency domain analysis (power spectra)
- Artifact detection rates
- Electrode impedance trends
- Beta-burst detection consistency

**Manual Review**:
- Visual inspection of signal traces (10% random sample)
- Burst detection validation
- Threshold calculation verification
- Cross-site signal comparison
- Anomaly investigation and resolution

#### Data Processing Validation
**Pipeline Verification**:
- Preprocessing step validation
- Feature extraction accuracy
- Classification model consistency
- Cross-site result comparison
- Statistical validation of outputs

---

## 7. Risk-Based Monitoring

### 7.1 Risk Assessment Framework

#### Risk Categories
**High Risk**:
- Primary endpoint data integrity
- Patient safety and adverse events
- Protocol compliance for key procedures
- Regulatory documentation completeness
- Cross-site data consistency

**Medium Risk**:
- Secondary endpoint data quality
- Training and certification compliance
- Equipment maintenance and calibration
- Data transfer and security
- Site performance metrics

**Low Risk**:
- Administrative documentation
- Non-critical protocol procedures
- Descriptive data elements
- Routine communication records

### 7.2 Risk Mitigation Strategies

#### Proactive Risk Management
**Real-Time Monitoring**: Immediate identification of quality issues
**Automated Alerts**: System-generated notifications for threshold breaches
**Preventive Training**: Regular refresher training for site staff
**Equipment Redundancy**: Backup systems and spare components
**Communication Protocols**: Clear escalation procedures for issues

#### Reactive Risk Response
**Issue Investigation**: Systematic root cause analysis
**Corrective Actions**: Specific measures to address identified problems
**Preventive Actions**: Changes to prevent recurrence
**Documentation**: Complete audit trail of all risk management activities
**Effectiveness Review**: Assessment of corrective action success

---

## 8. Data Management Workflows

### 8.1 Data Transfer Procedures

#### Site-to-Central Transfer
**Transfer Schedule**: Daily automated transfer at 02:00 local time
**Transfer Method**: Secure FTP with end-to-end encryption
**File Validation**: Checksum verification for all transferred files
**Backup Protocol**: Local backup maintained until transfer confirmation
**Error Handling**: Automatic retry with manual intervention escalation

#### Data Processing Workflow
```
Raw Data Receipt → File Integrity Check → Automated Processing → Quality Validation → Database Integration → Backup Creation
```

**Processing Steps**:
1. File format validation and conversion
2. Signal quality assessment
3. Beta-burst detection and feature extraction
4. Statistical validation and outlier detection
5. Database integration and indexing
6. Backup and archive procedures

### 8.2 Database Management

#### Database Structure
**Patient Data**: Demographics, medical history, assessments
**Session Data**: EEG parameters, technical metrics, outcomes
**Quality Data**: Signal quality, protocol compliance, deviations
**Administrative Data**: Staff logs, training records, communications
**Analysis Data**: Processed features, statistical results, reports

#### Data Integrity Measures
**Referential Integrity**: Foreign key constraints and relationship validation
**Data Validation**: Real-time checks during data entry
**Audit Trail**: Complete history of all data changes
**User Access Control**: Role-based permissions and authentication
**Backup and Recovery**: Regular backups with tested recovery procedures

---

## 9. Safety Monitoring

### 9.1 Adverse Event Monitoring

#### AE Classification System
**Severity Grading**:
- Grade 1 (Mild): Minimal symptoms, no intervention required
- Grade 2 (Moderate): Symptoms causing some interference with daily activities
- Grade 3 (Severe): Symptoms significantly interfering with daily activities
- Grade 4 (Life-threatening): Urgent intervention required
- Grade 5 (Death): Death related to adverse event

**Causality Assessment**:
- Definitely Related: Causal relationship certain
- Probably Related: Causal relationship likely
- Possibly Related: Causal relationship possible but uncertain
- Unlikely Related: Causal relationship doubtful
- Not Related: No causal relationship

#### Safety Reporting Procedures
**Serious Adverse Events (SAE)**:
- Immediate notification to PI within 24 hours
- Regulatory reporting within required timeframes
- Ethics committee notification per local requirements
- Documentation in safety database and CRF
- Follow-up until resolution or stabilization

### 9.2 Data Safety Monitoring Board (DSMB)

#### DSMB Composition
**Independent Statistician**: [Name] - Statistical oversight
**Clinical Expert**: [Name] - Clinical safety assessment
**Regulatory Expert**: [Name] - Regulatory compliance review
**Chair**: [Name] - Overall DSMB coordination

#### DSMB Responsibilities
**Safety Monitoring**: Regular review of adverse events and safety data
**Efficacy Monitoring**: Interim analysis and futility assessment
**Protocol Compliance**: Assessment of protocol adherence and modifications
**Recommendation Authority**: Authority to recommend study continuation, modification, or termination

---

## 10. Quality Assurance

### 10.1 Quality Management System

#### QMS Framework
**Quality Policy**: Commitment to highest standards of data quality and integrity
**Quality Procedures**: Standardized procedures for all critical processes
**Training Programs**: Comprehensive training for all personnel
**Quality Metrics**: Quantitative measures of quality performance
**Continuous Improvement**: Regular review and enhancement of quality systems

#### Quality Control Activities
**Internal Audits**: Regular internal reviews of procedures and compliance
**External Audits**: Regulatory or sponsor audits as required
**Management Reviews**: Quarterly QMS performance reviews
**Corrective Actions**: Systematic approach to addressing quality issues
**Preventive Actions**: Proactive measures to prevent quality problems

### 10.2 Standard Operating Procedures

#### SOP Management
**SOP Development**: Standardized format and approval process
**Version Control**: Systematic version control and change management
**Training Requirements**: Mandatory training on all applicable SOPs
**Compliance Monitoring**: Regular assessment of SOP adherence
**Update Procedures**: Regular review and update of all SOPs

---

## 11. Technology Infrastructure

### 11.1 IT System Requirements

#### Hardware Infrastructure
**Central Servers**: Redundant servers with failover capability
**Storage Systems**: High-availability storage with automated backup
**Network Infrastructure**: Secure, high-bandwidth network connections
**Workstations**: Validated computer systems for data analysis
**Backup Systems**: Separate geographic location for disaster recovery

#### Software Systems
**Database Management**: PostgreSQL with REDCap interface
**Data Processing**: MATLAB/Python environments for EEG analysis
**Statistical Software**: R, SAS, or equivalent for statistical analysis
**Security Software**: Antivirus, firewall, intrusion detection systems
**Backup Software**: Automated backup and recovery solutions

### 11.2 System Validation

#### Computer System Validation (CSV)
**Validation Planning**: Comprehensive validation plans for all systems
**Installation Qualification**: Verification of correct system installation
**Operational Qualification**: Verification of system functionality
**Performance Qualification**: Verification of system performance specifications
**Change Control**: Managed change process for system modifications

---

## 12. Regulatory Compliance

### 12.1 Regulatory Framework

#### Applicable Regulations
**ICH-GCP**: International Conference on Harmonisation Good Clinical Practice
**ISO 14155**: Clinical investigation of medical devices for human subjects
**FDA 21 CFR Part 820**: Quality system regulation for medical devices
**EU MDR**: Medical Device Regulation (2017/745)
**Local Regulations**: Country-specific requirements for all participating sites

#### Compliance Monitoring
**Regular Assessments**: Quarterly compliance reviews
**Training Programs**: Regular regulatory training for all personnel
**Documentation Reviews**: Systematic review of regulatory documentation
**Audit Readiness**: Continuous audit readiness assessment
**Corrective Actions**: Immediate response to compliance issues

### 12.2 Documentation Management

#### Master File Management
**Study Master File**: Comprehensive documentation of study conduct
**Investigator Site Files**: Site-specific regulatory documentation
**Version Control**: Systematic management of document versions
**Archive Procedures**: Secure archival of completed study documentation
**Retrieval Procedures**: Efficient retrieval for audits and inspections

---

## 13. Training and Personnel

### 13.1 Training Requirements

#### Initial Training Program
**GCP Training**: Good Clinical Practice certification required
**Protocol Training**: Comprehensive protocol and procedure training
**System Training**: Training on all study-specific systems and software
**Safety Training**: Adverse event recognition and reporting procedures
**Data Management Training**: CRF completion and data handling procedures

#### Ongoing Training
**Annual Refresher**: Annual updates on protocols and procedures
**System Updates**: Training on system changes and enhancements
**Regulatory Updates**: Training on new regulatory requirements
**Quality Improvement**: Training on lessons learned and best practices

### 13.2 Personnel Qualifications

#### Required Personnel
**Principal Investigator**: Medical degree with clinical research experience
**Study Coordinators**: Bachelor's degree with GCP certification
**EEG Technicians**: Certified EEG technician with training record
**Data Managers**: Bachelor's degree in relevant field with database experience
**Monitors**: Clinical research experience with monitoring certification

---

## 14. Timeline and Deliverables

### 14.1 Monitoring Timeline

#### Study Setup (Months 1-3)
- Site activation and initial monitoring visits
- Staff training and certification
- System validation and testing
- Baseline monitoring procedures establishment

#### Active Monitoring (Months 4-15)
- Monthly remote monitoring activities
- Quarterly on-site monitoring visits
- Continuous data quality monitoring
- Regular safety and compliance assessments

#### Study Completion (Months 16-18)
- Final data verification and database lock
- Close-out monitoring visits
- Final quality assessments
- Archive preparation and completion

### 14.2 Deliverables Schedule

**Monthly Deliverables**:
- Site monitoring reports
- Data quality summaries
- Safety reports
- Recruitment and retention updates

**Quarterly Deliverables**:
- Comprehensive study progress reports
- DSMB reports and recommendations
- Quality assurance assessments
- Regulatory compliance reviews

**Final Deliverables**:
- Final monitoring report
- Database validation report
- Quality assurance certification
- Regulatory submission support documentation

---

**Plan Status**: Ready for implementation
**Next Review**: Quarterly during study conduct
**Document Approval**: Principal Investigator and Quality Assurance approval required
**Effective Date**: Study initiation date