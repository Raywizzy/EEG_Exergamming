# Figure Legends - npj Parkinson's Disease Submission

## Main Figures

### Figure 1: Core15+ Biomarker Framework and Validation Design
**Panel A:** Schematic of Core15+ multi-feature biomarker panel showing integration of Core5 baseline features (beta-burst dynamics), spectral features (cross-frequency coupling, alpha metrics), connectivity features (phase locking index, coherence), and temporal features (burst synchronization). Each feature category contributes to ensemble classification.

**Panel B:** 3-site Leave-One-Site-Out (LOSO) validation design. Data from Iowa (ds004584, n=266), UCSD (ds002778, n=248), and UNM/Iowa (ds003490, n=50) datasets undergo locked preprocessing pipeline with CORAL domain adaptation for cross-site harmonization.

**Panel C:** Regulatory compliance framework showing locked processing parameters, cross-site external validation, statistical significance testing, and audit trail documentation meeting FDA/EMA digital biomarker guidelines.

---

### Figure 2: Phase V Optimization Results - Performance Progression
**Panel A:** Balanced accuracy progression across optimization phases. Baseline Phase IV (49.0% ± 2.8%, p = 0.60, not significant) → Core5 Enhanced (87.6% ± 10.4%, p = 0.036, significant) → Core15+ Optimized (97.2% ± 3.9%, p = 0.003, highly significant). Error bars show 95% confidence intervals.

**Panel B:** Statistical validation metrics. Forest plot showing effect sizes (Cohen's d) for each optimization phase: Phase IV (d = -0.35, small), Core5 Enhanced (d = 3.61, large), Core15+ (d = 12.0, very large). Dashed line indicates medium effect threshold (d = 0.5).

**Panel C:** Clinical threshold analysis. Horizontal bar chart showing performance relative to clinical deployment threshold (65% balanced accuracy). Phase IV below threshold (-16%), Core5 Enhanced exceeds (+22.6%), Core15+ substantially exceeds (+32.2%).

---

### Figure 3: Cross-Site Validation Performance and Feature Importance
**Panel A:** 3-site LOSO validation results. Bar chart showing balanced accuracy for each test site: Iowa (91.7%), UCSD (100.0%), UNM/Iowa (100.0%). Mean performance (97.2%) with 95% confidence interval [85.3%, 109.2%]. Red dashed line indicates clinical threshold (65%).

**Panel B:** Feature importance ranking from ensemble classifier. Horizontal bar plot showing top 10 most discriminative features: Spectral entropy (23%), Motor/posterior duty ratio (19%), Alpha peak frequency (16%), PLI motor-posterior (14%), Burst synchronization (12%), Mean duration (8%), Duty cycle (7%), Theta/beta ratio (6%), Coherence bilateral (5%), Temporal stability (4%).

**Panel C:** Cross-site feature stability analysis. Heatmap showing feature correlation coefficients across sites. High correlation (r > 0.8) indicates successful harmonization via CORAL domain adaptation.

---

## Supplementary Figures

### Figure S1: Dataset Characteristics and Preprocessing Pipeline
**Panel A:** Dataset overview showing recording parameters, subject demographics, and data quality metrics for each site. Table format with Iowa, UCSD, and UNM/Iowa characteristics.

**Panel B:** Preprocessing pipeline flowchart. Step-by-step illustration of locked preprocessing: raw EEG → bandpass filtering (1-40 Hz) → resampling (256 Hz) → common average reference → artifact rejection → beta band isolation (13-30 Hz) → burst detection → feature extraction.

**Panel C:** Quality control metrics. Distribution plots showing signal-to-noise ratios, data duration, and artifact rejection rates across all three sites.

---

### Figure S2: Ensemble Model Performance and Cross-Validation Analysis
**Panel A:** Individual classifier performance comparison. Bar chart showing balanced accuracy for Random Forest (89.3%), Logistic Regression (91.7%), and Voting Classifier (97.2%) in 3-site LOSO validation.

**Panel B:** Cross-validation stability analysis. Box plots showing balanced accuracy distributions across 5-fold cross-validation within each site. Median, quartiles, and outliers displayed for robustness assessment.

**Panel C:** Receiver Operating Characteristic (ROC) curves for each test site. Area Under Curve (AUC) values: Iowa (0.94), UCSD (1.00), UNM/Iowa (1.00). Average AUC = 0.98 indicating excellent discrimination.

---

## Figure Specifications

### Technical Requirements (Nature Portfolio Standards)
- **Resolution:** Minimum 300 DPI at final publication size
- **Format:** TIFF, EPS, or high-quality PDF
- **Color Mode:** RGB for online, CMYK for print
- **Font:** Arial or similar sans-serif, minimum 6pt
- **Line Width:** Minimum 0.5pt

### Size Specifications
- **Single Column:** 85mm width maximum
- **Double Column:** 175mm width maximum
- **Maximum Height:** 235mm
- **Multi-panel figures:** Clear panel labels (A, B, C) in bold

### Color Guidelines
- **Primary Colors:** Blue (#1f77b4), Orange (#ff7f0e), Green (#2ca02c)
- **Significance Indicators:** Red for p < 0.05, Dark red for p < 0.01
- **Threshold Lines:** Dashed red for clinical threshold (65%)
- **Confidence Intervals:** Gray shading or error bars

---

## Figure File Naming Convention

### Main Figures
- `Figure1_Core15_Framework.tiff`
- `Figure2_Performance_Progression.tiff`
- `Figure3_CrossSite_Validation.tiff`

### Supplementary Figures
- `FigureS1_Dataset_Preprocessing.tiff`
- `FigureS2_Ensemble_Performance.tiff`

---

## Statistical Annotations

### Significance Indicators
- **p < 0.05:** Single asterisk (*)
- **p < 0.01:** Double asterisk (**)
- **p < 0.001:** Triple asterisk (***)
- **Not significant:** "ns" or no marking

### Effect Size Indicators
- **Small effect (d ≥ 0.2):** Light shading
- **Medium effect (d ≥ 0.5):** Medium shading
- **Large effect (d ≥ 0.8):** Dark shading

### Performance Thresholds
- **Clinical threshold (65%):** Red dashed horizontal line
- **Chance performance (50%):** Black dashed horizontal line
- **Target exceeded:** Green shading above 65%

---

## Data Presentation Standards

### Error Bars
- **95% Confidence Intervals:** For all performance metrics
- **Standard Error:** Where appropriate for multiple measurements
- **Bootstrap Confidence Intervals:** For robust statistical validation

### Sample Size Reporting
- **Total N:** Always displayed (N = 564 total subjects)
- **Per-site N:** Iowa (n=266), UCSD (n=248), UNM/Iowa (n=50)
- **Per-group N:** Where applicable for balanced reporting

### Statistical Test Results
- **p-values:** Reported to 3 decimal places (p = 0.003)
- **Effect sizes:** Cohen's d to 1 decimal place (d = 12.0)
- **Confidence intervals:** Brackets format [85.3%, 109.2%]

---

## Figure Accessibility

### Color-blind Accessibility
- **Primary palette:** Colorbrewer-safe colors
- **Pattern fills:** Used in addition to colors
- **High contrast:** Minimum 3:1 ratio for text

### Alternative Text Descriptions
- **Figure 1:** "Core15+ framework schematic and validation design"
- **Figure 2:** "Performance progression from baseline to optimized biomarkers"
- **Figure 3:** "Cross-site validation results and feature importance"

---

*Figure legends prepared for npj Parkinson's Disease submission*
*All figures to be generated at publication-quality resolution*
*Legends follow Nature Portfolio formatting guidelines*