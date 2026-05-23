# Core15+ Feature Specification

**Version:** 1.0
**Target:** Clinical-grade EEG biomarkers for PD vs Control classification
**Baseline:** Core5 beta-burst features achieve 50.5% BA (chance level)
**Goal:** Achieve ≥65% balanced accuracy for clinical utility

---

## Feature Categories

### 1. Beta-Burst Features (Core5 Baseline)
*Retained from Phase 2 for comparison*

| Feature | Description | Units | Expected Range | Source |
|---------|-------------|-------|----------------|--------|
| `duration_cv` | Coefficient of variation of burst durations | unitless | 0.2-2.0 | Core5 |
| `duty_cycle` | Proportion of time in burst state | proportion | 0.01-0.3 | Core5 |
| `mean_duration_ms` | Average burst duration | milliseconds | 100-800 | Core5 |
| `median_duration_ms` | Median burst duration | milliseconds | 80-600 | Core5 |
| `motor_posterior_duty_ratio` | Motor/posterior duty cycle ratio | ratio | 0.5-1.5 | Core5 |

### 2. Spectral Power Features (New)
*Frequency domain analysis across multiple bands*

| Feature | Description | Units | Expected Range | Clinical Rationale |
|---------|-------------|-------|----------------|-------------------|
| `theta_power_rel` | Relative theta power (4-8 Hz) | proportion | 0.05-0.25 | PD cognitive changes |
| `alpha_power_rel` | Relative alpha power (8-12 Hz) | proportion | 0.15-0.45 | Motor cortex activity |
| `beta_power_rel` | Relative beta power (13-30 Hz) | proportion | 0.10-0.35 | Motor symptoms |
| `gamma_power_rel` | Relative gamma power (30-40 Hz) | proportion | 0.02-0.15 | Neural synchrony |
| `alpha_beta_ratio` | Alpha/beta power ratio | ratio | 0.5-3.0 | PD severity marker |
| `alpha_peak_freq` | Individual alpha frequency | Hz | 8.0-12.5 | Neural slowing |
| `beta_peak_freq` | Individual beta frequency | Hz | 15-25 | Beta peak shifts |

### 3. Temporal Burst Metrics (New)
*Advanced burst dynamics beyond Core5*

| Feature | Description | Units | Expected Range | Clinical Rationale |
|---------|-------------|-------|----------------|-------------------|
| `burst_rate` | Bursts per minute | bursts/min | 10-200 | PD motor patterns |
| `mean_burst_amplitude` | Average burst amplitude | μV² | 5-50 | Neural drive |
| `burst_amplitude_cv` | CV of burst amplitudes | unitless | 0.3-1.5 | Amplitude variability |
| `inter_burst_interval_mean` | Mean time between bursts | seconds | 0.5-10.0 | Rhythm disruption |
| `inter_burst_interval_cv` | CV of inter-burst intervals | unitless | 0.4-2.0 | Temporal irregularity |

### 4. Spatial Connectivity (New)
*Cross-regional interactions*

| Feature | Description | Units | Expected Range | Clinical Rationale |
|---------|-------------|-------|----------------|-------------------|
| `motor_coherence_beta` | C3-C4 beta coherence | coherence | 0.1-0.8 | Interhemispheric sync |
| `fronto_motor_beta_coh` | Fz-Cz beta coherence | coherence | 0.1-0.7 | Top-down control |
| `motor_posterior_alpha_coh` | C3-P3 alpha coherence | coherence | 0.1-0.6 | Sensorimotor loop |
| `hemispheric_asymmetry_beta` | (C3-C4)/(C3+C4) beta power | asymmetry | -0.3-0.3 | Lateralization |

### 5. Stability Metrics (New)
*Temporal consistency across recording*

| Feature | Description | Units | Expected Range | Clinical Rationale |
|---------|-------------|-------|----------------|-------------------|
| `alpha_power_stability` | CV of alpha power across windows | unitless | 0.1-0.8 | Neural stability |
| `beta_power_stability` | CV of beta power across windows | unitless | 0.1-0.8 | Motor consistency |
| `burst_rate_stability` | CV of burst rate across windows | unitless | 0.2-1.0 | Rhythmic stability |

---

## Implementation Details

### Channel Selection

**Motor Channels:** C3, Cz, C4
**Posterior Channels:** P3, Pz, P4, O1, Oz, O2
**Frontal Channels:** F3, Fz, F4
**Reference:** Linked ears or average reference (dataset-dependent)

### Frequency Bands

```python
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (30, 40)
}
```

### Window Parameters

- **Window length:** 4 seconds (2048 samples at 512 Hz)
- **Overlap:** 50% (2 seconds)
- **Minimum windows per subject:** 30 (2-minute recording minimum)
- **Artifact rejection:** ICA + automated threshold

### Quality Control Ranges

**Physiological Bounds:**
- Power features: 0.001-0.999 (relative)
- Coherence: 0.0-1.0
- Frequency peaks: Within band limits
- Temporal metrics: Positive, finite values

**Outlier Detection:**
- Per-feature: Mean ± 3×SD
- Multivariate: Mahalanobis distance > 3
- Missing data: Exclude subjects with >30% NaN features

---

## Expected Performance Improvement

### Hypotheses

1. **Spectral Power:** PD shows altered alpha/beta ratios → Better separation
2. **Connectivity:** Reduced motor coherence in PD → Enhanced discrimination
3. **Temporal Stability:** PD has more variable neural rhythms → Additional signal
4. **Multimodal Integration:** Combined features exceed single-domain performance

### Target Performance

| Metric | Core5 (Baseline) | Core15+ (Target) | Improvement |
|--------|------------------|------------------|-------------|
| **Balanced Accuracy** | 50.5% | **≥65%** | +14.5pp |
| **AUC** | ~0.6 | **≥0.75** | +0.15 |
| **Clinical Utility** | None | **Achieved** | Threshold met |

### Success Criteria

✅ **Primary:** Mean BA ≥ 65% across 3-site LOSO
✅ **Secondary:** At least 2/3 sites above 60% BA
✅ **Tertiary:** Statistically significant vs Core5 (paired t-test)

---

## Implementation Plan

### Phase 3.1: Core Implementation
1. `src/features/core15.py` - Feature extraction functions
2. Unit tests with synthetic signals
3. Validation against known EEG properties

### Phase 3.2: Dataset Processing
1. Extract Core15+ from all 3 labeled datasets
2. Quality control and outlier detection
3. Generate feature summary reports

### Phase 3.3: Validation
1. 3-site LOSO with Core15+ features
2. Statistical comparison vs Core5
3. Feature importance analysis

### Phase 3.4: Optimization
1. Feature selection (if needed)
2. Hyperparameter tuning
3. Ensemble methods

---

## Clinical Translation

### Regulatory Considerations
- **Feature interpretability:** All features have neurophysiological basis
- **Reproducibility:** Deterministic extraction pipeline
- **Validation:** Multi-site generalization demonstrated

### Deployment Requirements
- **Real-time capability:** Features extractable in <30 seconds
- **Minimal data:** Works with 2-minute recordings
- **Robustness:** Handles common artifacts automatically

---

*Specification version 1.0*
*Target implementation: Phase 3.1*
*Expected completion: Phase 3.4*