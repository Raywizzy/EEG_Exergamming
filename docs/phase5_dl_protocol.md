# Phase V Protocol: Deep Learning Enhancement
**Next-Generation EEG Biomarkers for Parkinson's Disease Assessment**

---

## Executive Summary

### Objective
Develop and validate deep learning models for EEG-based Parkinson's disease assessment, building upon the validated Core5 + CORAL foundation from Phase III-IV to achieve 70-80% balanced accuracy.

### Strategic Positioning
- **Foundation**: Core5 classical ML pipeline (60-65% BA, regulatory-ready)
- **Enhancement**: Deep learning models for improved performance
- **Validation**: Same LOSO framework, statistical rigor as Phase III
- **Regulatory**: Exploratory development for future AI/ML submission

### Key Innovation
End-to-end learning from raw EEG data while maintaining interpretability through hybrid architectures and comparison with validated Core5 biomarkers.

---

## Background & Rationale

### Phase III-IV Foundation
The Core5 + CORAL pipeline established:
- ✅ Multi-site validation across 3+ independent datasets
- ✅ Regulatory-grade statistical validation (LOSO + permutation testing)
- ✅ Clinical interpretability with β-burst biomarkers
- ✅ Performance baseline: 60-65% balanced accuracy

### Deep Learning Opportunity
**Performance Gap**: Classical ML plateau at ~65% leaves room for improvement
**Data Volume**: Combined datasets (1000+ sessions) enable deep learning
**Technical Advances**: EEGNet, temporal CNNs, and transformer architectures
**Clinical Need**: Higher accuracy for confident clinical decision support

### Regulatory Strategy
- **Phase V Status**: Research and development, not regulatory submission
- **Comparison Framework**: Deep learning vs Core5 baseline under identical conditions
- **Future Pathway**: AI/ML enhancement for v2.0 regulatory submission
- **Risk Management**: Maintain Core5 as interpretable, approved baseline

---

## Study Design

### Datasets & Sample Size
| Dataset | Sessions | Subjects | Conditions | Acquisition Details |
|---------|----------|----------|------------|-------------------|
| Leicester | 242 | 31 | OFF/ON medication | Clinical EEG feedback |
| MRC BNDU | 514 | 31 | REAL/SHAM stimulation | Neurofeedback training |
| UCSD ds002778 | ~300 | ~50 | Rest/task paradigms | High-density montage |
| **Total Phase V** | **~1000+** | **~110+** | **Binary classification** | **Multi-site diversity** |

**Additional Datasets (Optional):**
- PhysioNet EEG Motor Movement/Imagery
- Kaggle Parkinson's EEG collections
- OpenNeuro neurological datasets

### Validation Framework
**Primary Validation**: Leave-One-Site-Out (LOSO)
- Train on 2+ sites → Test on held-out site
- Identical to Phase III methodology for fair comparison
- Statistical validation: permutation tests, bootstrap CIs

**Secondary Validation**: Temporal splits within sites
- Train on early sessions → Test on later sessions
- Assess temporal generalization and model stability

**Tertiary Validation**: Subject-wise holdout
- Train on (n-k) subjects → Test on k held-out subjects
- Standard deep learning validation for architecture comparison

---

## Deep Learning Architectures

### 1. EEGNet (Baseline CNN)
**Reference**: Lawhern et al., 2018 - Compact CNN for EEG classification
**Architecture**:
```python
Input: Raw EEG (channels × time)
├── Temporal Conv (1×64, 125Hz/4 = ~32ms kernels)
├── Depthwise Conv (channels×1, spatial filtering)
├── Separable Conv (1×16, ~64ms kernels)
├── Global Average Pooling
└── Dense(2) → PD_REAL/PD_SHAM
```
**Advantages**: Lightweight, real-time compatible, proven on EEG
**Target Performance**: 65-70% BA (5-10% improvement over Core5)

### 2. Temporal CNN (β-burst Specialized)
**Custom Architecture**: Optimized for β-burst detection and classification
**Architecture**:
```python
Input: β-band envelope (13-30Hz Hilbert)
├── Multi-scale Conv1D (50ms, 100ms, 200ms kernels)
├── Spatial Attention (motor vs posterior regions)
├── Temporal Attention (burst vs non-burst periods)
├── Feature Fusion + Dropout
└── Classification Head
```
**Advantages**: Domain-specific, interpretable burst focus
**Target Performance**: 70-75% BA (β-burst pattern learning)

### 3. Recurrent Networks (Sequential Modeling)
**Architecture Options**:
- **Bidirectional LSTM**: Temporal context for burst sequences
- **GRU with Attention**: Efficient sequential processing
- **Transformer**: Self-attention for long-range dependencies

**Input Representations**:
- β-burst event sequences (onset, duration, amplitude)
- Sliding window β-power time series
- Multi-frequency spectrograms

**Target Performance**: 70-75% BA (temporal pattern recognition)

### 4. Hybrid CNN-RNN (Multi-Scale)
**Architecture**:
```python
Input: Raw EEG (channels × time × frequency)
├── Spatial CNN (channel interactions)
├── Temporal CNN (local patterns)
├── RNN Layer (sequence modeling)
├── Multi-Head Attention (cross-scale fusion)
└── Classification + Uncertainty Estimation
```
**Advantages**: Combines spatial, temporal, and sequential modeling
**Target Performance**: 75-80% BA (comprehensive feature learning)

---

## Technical Implementation

### Data Preprocessing Pipeline
```yaml
preprocessing:
  sampling_rate: 250  # Harmonized across sites
  filtering:
    notch: 50  # Power line noise
    bandpass: [1, 40]  # Broad EEG range
    beta_band: [13, 30]  # β-burst extraction

  referencing: CAR  # Common average reference

  segmentation:
    epoch_length: 2.0  # seconds
    overlap: 0.5  # 50% overlap for data augmentation

  quality_control:
    artifact_threshold: 4.0  # z-score
    bad_epoch_rate: 0.2  # max 20% rejection

  augmentation:  # For deep learning
    noise_injection: 0.1  # 10% noise variance
    temporal_jitter: 0.1  # ±100ms shifts
    amplitude_scaling: 0.2  # ±20% amplitude
```

### Model Training Protocol
```yaml
training:
  optimizer: AdamW
  learning_rate: 1e-3
  weight_decay: 1e-4

  scheduler: CosineAnnealingWarmRestarts
  warmup_epochs: 10
  max_epochs: 200

  batch_size: 32  # Subject-aware batching
  early_stopping: 20  # patience epochs

  loss_function: FocalLoss  # Class imbalance handling
  class_weights: balanced  # Auto-computed

  regularization:
    dropout: 0.3
    batch_norm: true
    gradient_clipping: 1.0
```

### LOSO Validation Implementation
```python
def loso_validation(datasets, model_class, config):
    """Leave-One-Site-Out validation for deep learning models"""

    results = {}
    for test_site in datasets.keys():
        # Combine training sites
        train_sites = [s for s in datasets.keys() if s != test_site]
        X_train, y_train = combine_sites(datasets, train_sites)
        X_test, y_test = datasets[test_site]

        # Site-wise stratified splits for hyperparameter tuning
        val_split = StratifiedGroupKFold(n_splits=3)

        # Hyperparameter optimization
        best_model = hyperparameter_search(
            model_class, X_train, y_train, val_split, config
        )

        # Final evaluation on test site
        predictions = best_model.predict(X_test)
        metrics = compute_metrics(y_test, predictions)

        results[test_site] = {
            'model': best_model,
            'metrics': metrics,
            'predictions': predictions
        }

    return aggregate_loso_results(results)
```

---

## Evaluation Framework

### Primary Metrics
| Metric | Target | Rationale |
|--------|--------|-----------|
| **Balanced Accuracy** | ≥70% | Primary endpoint, class-balanced |
| **ROC-AUC** | ≥0.75 | Discrimination capability |
| **Sensitivity** | ≥70% | PD detection rate |
| **Specificity** | ≥70% | False positive control |

### Secondary Metrics
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision/recall
- **Calibration**: Brier score, reliability diagrams
- **Uncertainty**: Prediction confidence analysis

### Comparison Framework
```python
model_comparison = {
    'Core5_Baseline': {
        'method': 'LogisticRegression + Core5 features',
        'performance': phase3_results,  # ~60-65% BA
        'interpretability': 'High',
        'deployment': 'Ready'
    },
    'EEGNet': {
        'method': 'Lightweight CNN',
        'target_performance': '65-70% BA',
        'interpretability': 'Medium',
        'deployment': 'Feasible'
    },
    'Hybrid_CNN_RNN': {
        'method': 'Multi-scale deep learning',
        'target_performance': '75-80% BA',
        'interpretability': 'Low',
        'deployment': 'Complex'
    }
}
```

### Statistical Validation
**Permutation Testing**: 1000 iterations with site-aware label shuffling
**Bootstrap Confidence Intervals**: 1000 samples for performance uncertainty
**McNemar's Test**: Paired comparison between Core5 and deep learning
**Friedman Test**: Multiple model comparison across sites

---

## Interpretability & Explainability

### Model Interpretation Techniques
1. **Attention Visualization**: Temporal and spatial attention weights
2. **Grad-CAM**: Gradient-based class activation mapping
3. **SHAP Values**: Feature importance for individual predictions
4. **Layer-wise Relevance Propagation**: Input attribution analysis

### Clinical Interpretability Framework
```python
interpretability_analysis = {
    'temporal_attention': {
        'method': 'Attention weight visualization',
        'output': 'Time periods relevant for classification',
        'clinical_relevance': 'Burst timing patterns'
    },
    'spatial_attention': {
        'method': 'Channel importance mapping',
        'output': 'Brain regions driving predictions',
        'clinical_relevance': 'Motor vs posterior contributions'
    },
    'feature_attribution': {
        'method': 'SHAP/LIME analysis',
        'output': 'Input features driving decisions',
        'clinical_relevance': 'Frequency bands and burst patterns'
    }
}
```

### Comparison with Core5 Biomarkers
- **Correlation Analysis**: Deep learning features vs Core5 biomarkers
- **Feature Importance**: Which learned features align with duration_cv, duty_cycle
- **Clinical Validation**: Do deep learning patterns match known β-burst physiology

---

## Risk Assessment & Mitigation

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-------------|
| **Overfitting to sites** | Medium | High | Aggressive regularization, LOSO validation |
| **Poor generalization** | Medium | High | Cross-site data augmentation, domain adaptation |
| **Computational complexity** | Low | Medium | Model compression, edge optimization |
| **Hyperparameter sensitivity** | Medium | Medium | Robust search, ensemble methods |

### Clinical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-------------|
| **Reduced interpretability** | High | Medium | Attention mechanisms, SHAP analysis |
| **Deployment complexity** | Medium | High | Maintain Core5 fallback, gradual rollout |
| **Regulatory uncertainty** | Medium | High | Exploratory positioning, future pathway |
| **Clinical acceptance** | Medium | Medium | Physician education, gradual integration |

---

## Success Criteria & Go/No-Go Gates

### Gate 1: Technical Feasibility (Month 2)
**Success Criteria:**
- [ ] EEGNet achieves ≥65% LOSO BA (5% improvement over Core5)
- [ ] Training pipeline stable across all sites
- [ ] Computational requirements feasible for deployment
- [ ] Basic interpretability analysis functional

**Go/No-Go Decision**: Proceed to advanced architectures if Gate 1 passed

### Gate 2: Performance Validation (Month 4)
**Success Criteria:**
- [ ] Best model achieves ≥70% LOSO BA (≥10% improvement)
- [ ] Statistical significance vs Core5 baseline (p < 0.05)
- [ ] Consistent performance across all test sites
- [ ] Interpretability analysis reveals clinically relevant patterns

**Go/No-Go Decision**: Proceed to full validation if Gate 2 passed

### Gate 3: Clinical Readiness (Month 6)
**Success Criteria:**
- [ ] Target performance achieved (70-80% BA)
- [ ] Complete statistical validation with confidence intervals
- [ ] Interpretability framework validated with clinical experts
- [ ] Deployment strategy defined with regulatory pathway

**Final Decision**: Recommend for Phase VI (Advanced AI Clinical Trial)

---

## Timeline & Milestones

### Months 1-2: Foundation Development
- **Week 1-2**: Data pipeline and preprocessing optimization
- **Week 3-4**: EEGNet implementation and initial validation
- **Week 5-6**: Temporal CNN development
- **Week 7-8**: Gate 1 evaluation and decision

### Months 3-4: Advanced Architecture Development
- **Week 9-10**: RNN/LSTM implementation
- **Week 11-12**: Hybrid CNN-RNN development
- **Week 13-14**: Hyperparameter optimization
- **Week 15-16**: Gate 2 evaluation and decision

### Months 5-6: Validation & Documentation
- **Week 17-18**: Complete LOSO validation
- **Week 19-20**: Statistical analysis and comparison
- **Week 21-22**: Interpretability analysis
- **Week 23-24**: Documentation and Gate 3 evaluation

---

## Deliverables

### Technical Deliverables
1. **Model Implementations**: PyTorch implementations of all architectures
2. **Training Pipeline**: Complete training, validation, and testing framework
3. **Evaluation Results**: LOSO performance across all models and sites
4. **Interpretability Analysis**: Attention maps, SHAP values, clinical insights

### Documentation Deliverables
1. **Phase V Results Report**: Comprehensive performance analysis
2. **Model Comparison Study**: Deep learning vs Core5 statistical comparison
3. **Clinical Interpretation Guide**: Explainability for medical professionals
4. **Deployment Recommendations**: Technical requirements and integration strategy

### Research Deliverables
1. **Manuscript Draft**: "Deep Learning Enhancement of EEG Biomarkers for Parkinson's Disease"
2. **Conference Presentations**: OHBM, SfN, EMBC submissions
3. **Patent Applications**: Novel architectures and domain adaptation techniques
4. **Open Source Release**: Model implementations and evaluation framework

---

## Future Pathways

### Phase VI: Advanced AI Clinical Trial (Optional)
- **Objective**: Prospective validation of best-performing deep learning model
- **Design**: Head-to-head comparison with Core5 in clinical setting
- **Timeline**: 12-18 months post-Phase V
- **Regulatory**: Prepare for AI/ML enhancement submission

### Platform Extension Opportunities
1. **Multi-Condition Validation**: Epilepsy, Alzheimer's, depression
2. **Multi-Modal Integration**: EEG + clinical data + imaging
3. **Real-Time Implementation**: Edge computing and mobile deployment
4. **Personalized Medicine**: Treatment response prediction and monitoring

---

## Conclusion

Phase V represents the next evolution in EEG biomarker development, building upon the solid foundation of validated Core5 classical ML to explore the full potential of deep learning for neurological assessment.

**Key Value Propositions:**
- **Performance Enhancement**: Target 70-80% BA vs 60-65% baseline
- **Technical Innovation**: End-to-end learning with clinical interpretability
- **Strategic Positioning**: Next-generation platform for AI-powered healthcare
- **Risk Management**: Maintains Core5 baseline while exploring advanced capabilities

**Success in Phase V will establish the complete technology stack for AI-powered neurological biomarkers, positioning for regulatory submission and clinical deployment of next-generation EEG assessment tools.**

---

**Protocol Version**: 1.0
**Date**: [Current Date]
**Status**: Ready for implementation post-Phase III completion
**Review Cycle**: Monthly progress reviews, quarterly gate evaluations