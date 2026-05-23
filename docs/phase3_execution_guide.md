# Phase III Execution Guide
**Quick Start: From 55.8% to 60%+ with Third Site Integration**

---

## 🚀 Quick Start: UCSD Integration

### Step 1: Download UCSD Dataset
```bash
# Option A: Auto-download (if openneuro-py available)
python scripts/phase3_ucsd_ingest.py --config config/phase3_ucsd.yaml --download

# Option B: Manual download
# 1. Go to https://openneuro.org/datasets/ds002778
# 2. Download dataset to: data/raw/ucsd_ds002778/
```

### Step 2: Process & Extract Features
```bash
# Harmonize EEG data (CAR, 50Hz notch, 250Hz resample)
python scripts/phase3_ucsd_ingest.py --config config/phase3_ucsd.yaml --process

# Extract Core5 beta burst features
python scripts/phase3_ucsd_ingest.py --config config/phase3_ucsd.yaml --features

# Or run all steps at once:
python scripts/phase3_ucsd_ingest.py --config config/phase3_ucsd.yaml --all
```

### Step 3: Run 3-Site LOSO Validation
```bash
# Once ucsd_Core5.csv exists, run Phase III evaluation
python scripts/phase3_loso_eval.py

# Expected output:
# - 3-site LOSO validation: Leicester, BNDU, UCSD
# - Target: ≥60% balanced accuracy
# - Statistical significance testing
# - Comprehensive performance report
```

---

## 📊 Expected Performance Trajectory

| Stage | Sites | Current BA | Target BA | Key Improvement |
|-------|-------|------------|-----------|-----------------|
| **Phase II** | 2 (MRC→Leicester) | 56.2% | — | CORAL baseline |
| **Phase III Baseline** | 2 (LOSO) | 55.8% | — | Framework validation |
| **Phase III Target** | 3 (LOSO) | — | **≥60%** | +Third site diversity |
| **Phase III Stretch** | 3 (Optimized) | — | **≥65%** | +Technical improvements |

### Key Success Drivers
1. **Sample Diversity**: UCSD adds different acquisition parameters
2. **CORAL Generalization**: Better domain adaptation with 3 sites
3. **Reduced Overfitting**: More robust cross-site validation
4. **Statistical Power**: Increased confidence in results

---

## 🔍 Quality Control Checkpoints

### Gate A: Data Integration (After Step 2)
```bash
# Verify UCSD data quality
ls results/features/ucsd_Core5.csv  # Should exist
head -5 results/features/ucsd_Core5.csv  # Check format

# Expected columns: subject_id, condition, site, duration_cv, duty_cycle,
# mean_duration_ms, median_duration_ms, motor_posterior_duty_ratio
```

### Gate B: LOSO Performance (After Step 3)
**Success Criteria:**
- [ ] Mean LOSO BA ≥ 60%
- [ ] 95% CI lower bound ≥ 55%
- [ ] All sites ≥ 55% minimum BA
- [ ] Statistical significance p < 0.05

---

## 🎯 Acceptance Criteria

### Primary Endpoint: Cross-Site Generalization
- **Metric**: Balanced accuracy in 3-site LOSO
- **Target**: ≥60% mean performance
- **Significance**: p < 0.05 (permutation test)
- **Consistency**: No site <55% BA

### Secondary Endpoints
- **Sensitivity**: ≥60% per site
- **Specificity**: ≥60% per site
- **ROC-AUC**: ≥0.65 per site
- **CORAL Improvement**: ≥5% vs baseline transfer

---

## 🔧 Troubleshooting

### Issue: "No EEG files found"
**Solution**: Check dataset structure
```bash
find data/raw/ucsd_ds002778 -name "*.edf" -o -name "*.fif" -o -name "*.set" | head -5
```

### Issue: "Missing motor/posterior channels"
**Solution**: Check channel mapping
```bash
# Edit config/phase3_ucsd.yaml to match available channels
# Common alternatives: FCz for Cz, CP3/CP4 for C3/C4
```

### Issue: "Performance below 60%"
**Options**:
1. **Optimize hyperparameters**: Grid search for each LOSO split
2. **Advanced domain adaptation**: Try DANN or MMD methods
3. **Feature engineering**: Explore Core3, Core7 variants
4. **Ensemble methods**: Combine multiple CORAL transforms

---

## 📋 Deliverables Checklist

### Technical Deliverables
- [ ] `results/features/ucsd_Core5.csv` - UCSD feature table
- [ ] `results/phase3_LOSO/loso_results_*.json` - Complete LOSO results
- [ ] `results/phase3_LOSO/loso_summary_*.md` - Performance summary
- [ ] `data/harmonized/ucsd/` - Preprocessed UCSD data

### Performance Metrics
- [ ] 3-site LOSO balanced accuracy table
- [ ] Per-site confusion matrices and ROC curves
- [ ] Bootstrap confidence intervals
- [ ] Permutation test p-values
- [ ] CORAL alignment effectiveness analysis

### Documentation
- [ ] Error analysis and failure mode investigation
- [ ] Site-specific performance breakdowns
- [ ] Subgroup analysis (age, disease duration)
- [ ] Technical optimization recommendations

---

## 🚀 Phase IV Preview

### Upon Phase III Success (≥60% LOSO BA):
1. **Regulatory Submission**: FDA Pre-Sub meeting preparation
2. **Clinical Trial Design**: Prospective validation study (n=200+)
3. **Commercial Development**: Real-time CDSS system
4. **Technology Enhancement**: Deep learning pilot comparison

### Next Immediate Actions:
- Compile Phase III evidence package
- Draft Phase IV clinical trial protocol
- Engage regulatory consultants
- Establish clinical site partnerships

---

## 📞 Support & Next Steps

### If Phase III Succeeds:
**🎉 Congratulations!** You've achieved regulatory-grade multi-site validation.
- Proceed to regulatory documentation
- Begin Phase IV clinical trial design
- Consider deep learning enhancement pilot

### If Performance Falls Short:
**🔧 Optimization Required:**
- Analyze failure modes and site-specific issues
- Implement advanced domain adaptation techniques
- Consider additional site integration or feature engineering
- Reassess acceptance criteria and regulatory pathway

---

**Phase III Status**: ⚡ READY TO EXECUTE
**Next Milestone**: 3-site LOSO ≥60% BA achievement
**Ultimate Goal**: FDA-cleared EEG biomarker for Parkinson's disease