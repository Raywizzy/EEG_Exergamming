# Figure Captions for NPJ Digital Medicine Submission

## Figure 1: Framework Architecture
**Caption:** Schematic overview of the self-monitoring EEG biomarker validation framework. The system integrates (A) locked preprocessing pipeline with standardized parameters, (B) Core5 beta-burst feature extraction, (C) CORAL domain adaptation for cross-site harmonization, (D) LOSO validation design, and (E) automated monitoring with traffic-light badge system. Real-time performance tracking enables immediate detection of optimization needs.

## Figure 2: Cross-Site LOSO Validation Results
**Caption:** Leave-one-site-out cross-validation results across three independent datasets. (A) Balanced accuracy by test site showing consistent performance near chance levels. (B) Confidence intervals (95%) for each validation fold. (C) Statistical significance testing results (t-test vs 50% chance). Error bars represent standard error of the mean. No significant difference from chance performance was observed (p = 0.60).

## Figure 3: Traffic-Light Badge System
**Caption:** Automated performance monitoring via traffic-light badge system. (A) Real-time badge generation showing current validation status. (B) Badge color coding: Green ≥65% (clinical threshold), Orange 50-65% (above chance), Red ≤50% (chance level). (C) Integration with automated reporting pipeline. Badges update automatically following each validation run, enabling immediate performance assessment.

## Figure 4: Cross-Site Harmonization via CORAL
**Caption:** CORAL domain adaptation for cross-site harmonization. (A) Feature distributions before harmonization showing site-specific differences. (B) Aligned feature distributions after CORAL application. (C) Quantitative assessment of harmonization effectiveness via distribution overlap metrics. CORAL successfully reduced cross-site variability while preserving within-site structure.

## Figure 5: Infrastructure Scalability and Performance
**Caption:** Scalability analysis of the validation framework. (A) Processing time per subject across dataset sizes. (B) Memory usage during validation runs. (C) Badge generation latency following validation completion. The framework demonstrates excellent scalability with sub-second per-subject processing and real-time monitoring capabilities.

## Figure 6: Regulatory Compliance Dashboard
**Caption:** Automated regulatory compliance reporting dashboard. (A) Locked pipeline verification showing parameter consistency. (B) Cross-site validation confirmation with complete data separation. (C) Statistical power assessment with appropriate degrees of freedom. (D) Audit trail documentation with version control integration. Dashboard auto-generates following each validation run.

## Supplementary Figure S1: Dataset Characteristics
**Caption:** Comprehensive overview of the three validation datasets. (A) Subject demographics and clinical characteristics. (B) Recording parameters and equipment specifications. (C) Data quality metrics and preprocessing outcomes. (D) Feature distribution summaries across sites. All datasets met inclusion criteria for multi-site validation.

## Supplementary Figure S2: Feature Extraction Validation
**Caption:** Validation of Core5 beta-burst feature extraction. (A) Example beta-burst detection on representative EEG traces. (B) Burst duration distributions across sites. (C) Duty cycle validation against physiological ranges. (D) Motor/posterior ratio consistency checks. All extracted features fell within expected physiological ranges.

---

**Figure Preparation Guidelines:**
- All figures prepared at ≥300 DPI resolution
- Color schemes optimized for accessibility (colorblind-friendly)
- Text size ≥8pt for readability in print
- Consistent styling across all panels
- Vector graphics (SVG) converted to high-resolution raster formats
