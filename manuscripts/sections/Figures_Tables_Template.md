# Figures & Tables - Biomarker Validation Template

**Publication-Ready Figure Captions and Table Templates**

---

## FIGURES

### Figure 1: Study Design and Workflow
**Caption:** Multi-site validation framework for Core5 EEG biomarkers. **(A)** CONSORT-style participant flow diagram showing inclusion/exclusion across three independent sites: Iowa (ds004584), UCSD (ds002778), and UNM/Iowa (ds003490). **(B)** Leave-One-Site-Out (LOSO) cross-validation strategy with training/testing splits. **(C)** Automated processing pipeline from raw EEG through Core5 feature extraction, CORAL domain adaptation, and classification with real-time quality control monitoring. **Total N = [TOTAL_SUBJECTS]** ([TOTAL_PD] PD, [TOTAL_HC] HC).

---

### Figure 2: Multi-Site Validation Results
**Caption:** Cross-site validation performance of Core5 biomarkers. **(A)** Leave-One-Site-Out (LOSO) balanced accuracy results by fold. Bars show individual site performance when used as test set, with error bars representing 95% confidence intervals. Horizontal dashed line indicates clinical threshold (65% BA). Mean performance: **[MEAN_BA]%** ± **[CI_WIDTH]%** CI. **(B)** Statistical significance testing: one-sample t-test vs chance (50%) showing t(**[DF]**) = **[T_STAT]**, p = **[P_VALUE]**. Effect size Cohen's d = **[COHENS_D]** (**[EFFECT_SIZE_INTERPRETATION]**). **(C)** Receiver Operating Characteristic (ROC) curves for each LOSO fold with area under curve (AUC) values. Mean AUC = **[MEAN_AUC]** ± **[AUC_CI]%**.

---

### Figure 3: Core5 Biomarker Distributions
**Caption:** Cross-site consistency of Core5 beta-burst features. Violin plots comparing PD (red) vs healthy controls (blue) across all three sites for: **(A)** Duration coefficient of variation, **(B)** Duty cycle (%), **(C)** Mean duration (ms), **(D)** Median duration (ms), and **(E)** Motor/posterior duty cycle ratio. Overlaid points show individual subject values. Statistical significance assessed via Wilcoxon rank-sum tests: **[FEATURE_STATS]**. Site consistency demonstrated by overlapping distributions and consistent effect directions across Iowa, UCSD, and UNM/Iowa datasets.

---

### Figure 4: Domain Adaptation Effectiveness
**Caption:** CORAL domain adaptation for cross-site harmonization. **(A)** Pre-adaptation: t-SNE visualization of Core5 feature space colored by acquisition site, showing clear site clustering. **(B)** Post-adaptation: CORAL-transformed feature space with reduced site effects while preserving PD/HC separability. **(C)** Quantitative assessment: reduction in site-wise feature distribution differences measured by maximum mean discrepancy (MMD) before vs after CORAL adaptation. **[ADAPTATION_EFFECTIVENESS]** reduction in cross-site variability achieved.

---

### Figure 5: Real-Time Monitoring Dashboard
**Caption:** Automated quality control and performance monitoring system. **(A)** Traffic-light badge system showing real-time validation status: GREEN (≥65% BA), ORANGE (50-65% BA), RED (<50% BA). Current status: **[BADGE_COLOR]**. **(B)** Processing efficiency metrics: mean processing time **[PROCESSING_TIME]** seconds per subject, **[QC_PASS_RATE]%** quality control pass rate. **(C)** Auto-updating manuscript integration showing synchronized results across analysis outputs, ensuring reproducibility and real-time documentation.

---

## TABLES

### Table 1: Dataset Characteristics
**Caption:** Demographic and acquisition characteristics across validation sites.

| **Characteristic** | **Iowa (ds004584)** | **UCSD (ds002778)** | **UNM/Iowa (ds003490)** | **Total** |
|-------------------|---------------------|---------------------|-------------------------|-----------|
| **Participants (N)** | **[IOWA_N_TOTAL]** | **[UCSD_N_TOTAL]** | **[DS003490_N_TOTAL]** | **[TOTAL_N]** |
| PD | **[IOWA_N_PD]** | **[UCSD_N_PD]** | **[DS003490_N_PD]** | **[TOTAL_PD]** |
| Healthy Controls | **[IOWA_N_HC]** | **[UCSD_N_HC]** | **[DS003490_N_HC]** | **[TOTAL_HC]** |
| **Age (years)** | | | | |
| PD | **[IOWA_AGE_PD]** ± **[IOWA_AGE_PD_SD]** | **[UCSD_AGE_PD]** ± **[UCSD_AGE_PD_SD]** | **[DS003490_AGE_PD]** ± **[DS003490_AGE_PD_SD]** | **[TOTAL_AGE_PD]** ± **[TOTAL_AGE_PD_SD]** |
| HC | **[IOWA_AGE_HC]** ± **[IOWA_AGE_HC_SD]** | **[UCSD_AGE_HC]** ± **[UCSD_AGE_HC_SD]** | **[DS003490_AGE_HC]** ± **[DS003490_AGE_HC_SD]** | **[TOTAL_AGE_HC]** ± **[TOTAL_AGE_HC_SD]** |
| **Sex (% Female)** | | | | |
| PD | **[IOWA_SEX_PD]%** | **[UCSD_SEX_PD]%** | **[DS003490_SEX_PD]%** | **[TOTAL_SEX_PD]%** |
| HC | **[IOWA_SEX_HC]%** | **[UCSD_SEX_HC]%** | **[DS003490_SEX_HC]%** | **[TOTAL_SEX_HC]%** |
| **EEG System** | **[IOWA_EEG_SYSTEM]** | **[UCSD_EEG_SYSTEM]** | **[DS003490_EEG_SYSTEM]** | - |
| **Sampling Rate (Hz)** | **[IOWA_SRATE]** | **[UCSD_SRATE]** | **[DS003490_SRATE]** | - |
| **Channel Count** | **[IOWA_CHANNELS]** | **[UCSD_CHANNELS]** | **[DS003490_CHANNELS]** | - |
| **Recording Duration (min)** | **[IOWA_DURATION]** | **[UCSD_DURATION]** | **[DS003490_DURATION]** | - |

---

### Table 2: Cross-Site Validation Performance
**Caption:** Leave-One-Site-Out (LOSO) classification performance with 95% confidence intervals.

| **Test Site** | **Training Sites** | **N (PD/HC)** | **Balanced Accuracy** | **Sensitivity** | **Specificity** | **PPV** | **NPV** | **AUC** |
|---------------|-------------------|---------------|----------------------|----------------|----------------|---------|---------|---------|
| **Iowa** | UCSD + ds003490 | **[IOWA_N_PD]**/[IOWA_N_HC] | **[IOWA_BA]%** (**[IOWA_BA_CI]**) | **[IOWA_SENS]%** (**[IOWA_SENS_CI]**) | **[IOWA_SPEC]%** (**[IOWA_SPEC_CI]**) | **[IOWA_PPV]%** | **[IOWA_NPV]%** | **[IOWA_AUC]** |
| **UCSD** | Iowa + ds003490 | **[UCSD_N_PD]**/[UCSD_N_HC] | **[UCSD_BA]%** (**[UCSD_BA_CI]**) | **[UCSD_SENS]%** (**[UCSD_SENS_CI]**) | **[UCSD_SPEC]%** (**[UCSD_SPEC_CI]**) | **[UCSD_PPV]%** | **[UCSD_NPV]%** | **[UCSD_AUC]** |
| **ds003490** | Iowa + UCSD | **[DS003490_N_PD]**/[DS003490_N_HC] | **[DS003490_BA]%** (**[DS003490_BA_CI]**) | **[DS003490_SENS]%** (**[DS003490_SENS_CI]**) | **[DS003490_SPEC]%** (**[DS003490_SPEC_CI]**) | **[DS003490_PPV]%** | **[DS003490_NPV]%** | **[DS003490_AUC]** |
| **Mean ± SD** | - | **[TOTAL_N_PD]**/[TOTAL_N_HC] | **[MEAN_BA]** ± **[SD_BA]%** | **[MEAN_SENS]** ± **[SD_SENS]%** | **[MEAN_SPEC]** ± **[SD_SPEC]%** | **[MEAN_PPV]** ± **[SD_PPV]%** | **[MEAN_NPV]** ± **[SD_NPV]%** | **[MEAN_AUC]** ± **[SD_AUC]** |

**Statistical Analysis:** One-sample t-test vs 50% chance: t(**[DF]**) = **[T_STAT]**, p = **[P_VALUE]** (one-sided). Effect size: Cohen's d = **[COHENS_D]** (**[EFFECT_SIZE_INTERPRETATION]** effect). **PPV** = Positive Predictive Value, **NPV** = Negative Predictive Value, **AUC** = Area Under ROC Curve.

---

### Table 3: Core5 Feature Performance
**Caption:** Individual biomarker discriminative performance across sites.

| **Core5 Feature** | **Effect Size (Cohen's d)** | **Statistical Significance** | **Clinical Interpretation** |
|-------------------|----------------------------|------------------------------|----------------------------|
| **Duration CV** | **[DURATION_CV_COHENS_D]** (**[DURATION_CV_CI]**) | p = **[DURATION_CV_P]** | **[DURATION_CV_INTERPRETATION]** |
| **Duty Cycle** | **[DUTY_CYCLE_COHENS_D]** (**[DUTY_CYCLE_CI]**) | p = **[DUTY_CYCLE_P]** | **[DUTY_CYCLE_INTERPRETATION]** |
| **Mean Duration** | **[MEAN_DUR_COHENS_D]** (**[MEAN_DUR_CI]**) | p = **[MEAN_DUR_P]** | **[MEAN_DUR_INTERPRETATION]** |
| **Median Duration** | **[MEDIAN_DUR_COHENS_D]** (**[MEDIAN_DUR_CI]**) | p = **[MEDIAN_DUR_P]** | **[MEDIAN_DUR_INTERPRETATION]** |
| **Motor/Posterior Ratio** | **[RATIO_COHENS_D]** (**[RATIO_CI]**) | p = **[RATIO_P]** | **[RATIO_INTERPRETATION]** |

Effect sizes calculated across all sites using pooled standard deviations. Statistical significance assessed via Wilcoxon rank-sum tests with Bonferroni correction for multiple comparisons (α = 0.01).

---

### Table 4: Processing Efficiency and Quality Control
**Caption:** Computational performance and quality metrics across validation pipeline.

| **Metric** | **Iowa** | **UCSD** | **ds003490** | **Overall** |
|------------|----------|----------|--------------|-------------|
| **Processing Time** | | | | |
| Mean (seconds/subject) | **[IOWA_PROC_TIME]** | **[UCSD_PROC_TIME]** | **[DS003490_PROC_TIME]** | **[MEAN_PROC_TIME]** |
| Range | **[IOWA_PROC_RANGE]** | **[UCSD_PROC_RANGE]** | **[DS003490_PROC_RANGE]** | **[OVERALL_PROC_RANGE]** |
| **Quality Control** | | | | |
| QC Pass Rate (%) | **[IOWA_QC_PASS]%** | **[UCSD_QC_PASS]%** | **[DS003490_QC_PASS]%** | **[OVERALL_QC_PASS]%** |
| Artifact Rejection (%) | **[IOWA_ARTIFACT]%** | **[UCSD_ARTIFACT]%** | **[DS003490_ARTIFACT]%** | **[OVERALL_ARTIFACT]%** |
| **Resource Usage** | | | | |
| Peak Memory (GB) | **[IOWA_MEMORY]** | **[UCSD_MEMORY]** | **[DS003490_MEMORY]** | **[MAX_MEMORY]** |
| CPU Utilization (%) | **[IOWA_CPU]%** | **[UCSD_CPU]%** | **[DS003490_CPU]%** | **[MEAN_CPU]%** |

Processing performed on **[HARDWARE_SPECS]** hardware. All processing completed successfully with **[SUCCESS_RATE]%** success rate across **[TOTAL_SUBJECTS]** subjects.

---

## SUPPLEMENTARY FIGURES

### Supplementary Figure S1: Beta-Burst Detection Algorithm
**Caption:** Validation of median + 2×MAD beta-burst detection threshold across sites. **(A)** Example beta-band (13-30 Hz) amplitude time series from representative PD and HC subjects. **(B)** Threshold calculation and burst detection. **(C)** Cross-site threshold consistency and burst rate validation.

### Supplementary Figure S2: Feature Distribution Quality Control
**Caption:** Quality control assessment of Core5 features. **(A)** Feature value distributions across sites before CORAL adaptation. **(B)** Outlier detection and removal criteria. **(C)** Post-processing feature distributions showing harmonization effectiveness.

### Supplementary Figure S3: Classification Algorithm Comparison
**Caption:** Performance comparison of Core5 + Logistic Regression vs alternative approaches. Balanced accuracy comparison across LOSO folds for: **(A)** Random Forest, **(B)** SVM (RBF kernel), **(C)** Gradient Boosting, **(D)** Deep Neural Network. Core5 + LR performance: **[CORE5_LR_PERFORMANCE]**.

---

**Auto-Population Status:** All **[PLACEHOLDER]** fields ready for automatic population when ds003490 processing completes. Figure generation scripts prepared for immediate plot creation with actual data.