# DATA SOURCES

## Approved Datasets for EEG Exergaming Project

This document lists the ONLY approved datasets for this project. Any use of synthetic, fabricated, or placeholder data is strictly forbidden.

### Primary Training Dataset (PERFECT MATCH!)

**🎯 MRC BNDU Neurofeedback Dataset** - EXACTLY what we need!

**Dataset Name**: "LFPs and EEGs from patients with Parkinson's disease during neurofeedback training"
**DOI**: 10.60964/bndu-4jde-7j28
**URL**: https://data.mrc.ox.ac.uk/data-set/lfps-and-eegs-patients-parkinsons-disease-during-neurofeedback-training
**License**: CC BY-SA 4.0 (Open access)

**Why This is Perfect**:
- ✅ **12 PD patients** (4 females) - no healthy controls
- ✅ **Real vs Sham conditions**:
  - **PD_REAL** = 'Training' condition (40 trials with real neurofeedback)
  - **PD_SHAM** = 'No Training' condition (40 trials without feedback)
- ✅ **EEG + LFP data** from subthalamic nucleus (STN)
- ✅ **Task-based**: Sequential neurofeedback-behavior task
- ✅ **Beta-targeted**: Perfect for alpha/beta band analysis
- ✅ **Published**: eLife 2020 (Tan et al.) - validated methodology

**Experimental Design**:
- 30s rest + 10 trials Training + 10 trials No Training per session
- 80 trials total per subject (40 Training, 40 No Training)
- Real-time beta burst detection in STN LFPs
- Motor initiation task with reaction time measurements

### Supplementary Datasets (For Feature Robustness)

1. **OpenNeuro ds002778** - UC San Diego PD EEG (resting-state)
   - Use: Feature distribution expansion, pretraining
   - Filter: PD subjects only

2. **OpenNeuro ds003490** - PD cognitive/motor task EEG
   - Use: Task-related feature validation
   - Filter: PD subjects only

3. **OpenNeuro ds004584** - Multi-site PD EEG
   - Use: Cross-site robustness testing
   - Filter: PD subjects only

### External Validation Dataset

**Dataset Name**: University of Leicester EEG Exergaming Dataset
**Status**: ✅ AVAILABLE - Located at data/external/leicester_dataset/
**Usage**: EXTERNAL VALIDATION ONLY - Never for training
**Description**: EEG recordings from patients performing exergaming trials
**Signal Type**: EEG (non-invasive scalp recordings)
**Subjects**: 17 subjects (A1-A16, AI)
**Sessions**: 250 unique sessions
**Files**: 1,232 CSV + 1,105 pickle files
**Date Range**: 2024-05-01 to 2024-12-12

### Classification Requirements

**ONLY These Classes Allowed**:
- **PD_REAL**: Parkinson's patients with real EEG feedback
- **PD_SHAM**: Parkinson's patients with sham/random feedback

**PROHIBITED Classes**:
- ❌ CONTROL: Healthy control subjects not allowed
- ❌ HC: Healthy controls not allowed
- ❌ Any healthy subject comparisons

### Data Validation Checklist

Before using any dataset, verify:
- [ ] Dataset is from approved public source
- [ ] No healthy control subjects included in training
- [ ] Only PD_REAL vs PD_SHAM classification
- [ ] Leicester dataset only in data/external/
- [ ] No NaN or UNKNOWN values in class labels
- [ ] Subject IDs consistent across sessions
- [ ] Sampling rates documented

### Prohibited Data Sources

The following are STRICTLY FORBIDDEN:
- Synthetic EEG data
- Simulated signals
- Generated or fabricated datasets
- Placeholder data from tutorials
- Private/proprietary datasets
- **Healthy control subjects in training data**

### Dataset Access Priority

1. **First Priority**: Find public PD dataset with real vs sham feedback
2. **Second Priority**: Adapt existing PD EEG datasets to real/sham paradigm
3. **Last Resort**: Use motor task datasets and simulate feedback conditions

### Data Access Log

All data access must be logged with:
- Timestamp (ISO format)
- Dataset source and DOI
- Files accessed
- Purpose of access
- Compliance verification

**Last Updated**: 2025-09-13T18:52:00.000Z
**Next Review**: After primary dataset selection