# Phase V Execution Guide: Deep Learning Enhancement
**From 65% to 80% Balanced Accuracy with Next-Generation EEG Biomarkers**

---

## 🚀 Quick Start Overview

### Phase V Objectives
- **Performance Target**: 70-80% balanced accuracy (vs 60-65% Core5 baseline)
- **Technical Goal**: End-to-end learning from raw EEG data
- **Validation Method**: Same LOSO framework as Phase III for fair comparison
- **Strategic Position**: Next-generation AI enhancement for v2.0 regulatory submission

### Prerequisites
- ✅ **Phase III Completed**: Core5 + CORAL baseline established (60-65% BA)
- ✅ **Multi-Site Data**: Leicester + BNDU + UCSD datasets available
- ✅ **Computing Resources**: GPU access recommended (CUDA compatible)
- ✅ **Deep Learning Stack**: PyTorch, Lightning, Optuna for hyperparameter optimization

---

## 📋 **PHASE V EXECUTION CHECKLIST**

### **Stage 1: Environment Setup (Week 1)**

#### Install Deep Learning Dependencies
```bash
# Core deep learning stack
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning torchmetrics
pip install optuna wandb  # Hyperparameter optimization and experiment tracking

# EEG-specific libraries
pip install mne scikit-learn scipy
pip install captum  # Model interpretability
pip install seaborn matplotlib plotly  # Visualization

# Optional: Specialized EEG deep learning
pip install braindecode  # EEG deep learning toolkit
```

#### Verify GPU Access
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

#### Create Phase V Directory Structure
```bash
mkdir -p {data/phase5,models/phase5,results/phase5,logs/phase5}
mkdir -p {src/models,src/data,src/training,src/evaluation}
mkdir -p results/phase5/{eegnet,temporal_cnn,bilstm,hybrid}
```

### **Stage 2: Data Preparation (Week 1-2)**

#### Unified Dataset Preparation
```bash
# Create unified Phase V dataset
python scripts/phase5_prepare_data.py --config config/phase5_datasets.yaml

# Expected output:
# - data/phase5/leicester_processed.h5
# - data/phase5/bndu_processed.h5
# - data/phase5/ucsd_processed.h5
# - data/phase5/combined_metadata.csv
```

#### Data Quality Verification
```bash
# Verify data integrity and balance
python scripts/phase5_data_qc.py --data_dir data/phase5/

# Check outputs:
# - Data distribution across sites
# - Class balance verification
# - Signal quality metrics
# - Channel alignment confirmation
```

### **Stage 3: Baseline Implementation (Week 2-3)**

#### Implement EEGNet (Entry Point)
```bash
# Start with lightweight EEGNet
python scripts/phase5_train_eegnet.py --config config/phase5_models.yaml

# Monitor training progress
tensorboard --logdir logs/phase5/eegnet/

# Expected performance: 65-70% LOSO BA
```

#### Core5 Comparison Baseline
```bash
# Reproduce Phase III Core5 results for comparison
python scripts/phase5_core5_baseline.py --config config/phase5_models.yaml

# This ensures fair comparison using identical data splits
```

### **Stage 4: Advanced Architectures (Week 3-5)**

#### Temporal CNN (β-burst Specialized)
```bash
# Train temporal CNN on β-burst patterns
python scripts/phase5_train_temporal_cnn.py --config config/phase5_models.yaml

# Target: 70-75% BA with burst-specific learning
```

#### Bidirectional LSTM
```bash
# Sequential modeling approach
python scripts/phase5_train_bilstm.py --config config/phase5_models.yaml

# Focus on temporal dependencies in burst sequences
```

#### Hybrid CNN-RNN
```bash
# Most sophisticated architecture
python scripts/phase5_train_hybrid.py --config config/phase5_models.yaml

# Target: 75-80% BA with multi-scale learning
```

### **Stage 5: Hyperparameter Optimization (Week 4-5)**

#### Automated Hyperparameter Search
```bash
# Use Optuna for systematic optimization
python scripts/phase5_hyperopt.py \
  --model eegnet \
  --n_trials 100 \
  --timeout 3600

# Run for each architecture:
# - EEGNet: Focus on kernel sizes, filters
# - Temporal CNN: Attention parameters
# - BiLSTM: Hidden sizes, sequence length
# - Hybrid: Multi-branch fusion weights
```

### **Stage 6: LOSO Validation (Week 5-6)**

#### Complete Multi-Site Validation
```bash
# Run LOSO validation for all models
python scripts/phase5_loso_validation.py \
  --config config/phase5_models.yaml \
  --models eegnet,temporal_cnn,bilstm,hybrid

# Outputs:
# - results/phase5/loso_results.json
# - results/phase5/model_comparison.json
# - results/phase5/statistical_validation.json
```

#### Statistical Comparison Framework
```bash
# Compare against Core5 baseline
python scripts/phase5_statistical_comparison.py \
  --baseline core5 \
  --deep_learning_results results/phase5/loso_results.json

# Statistical tests:
# - McNemar's test for paired comparisons
# - Friedman test for multiple models
# - Bootstrap confidence intervals
```

---

## 📊 **PERFORMANCE TRACKING & METRICS**

### Primary Evaluation Metrics

| Model | Target BA | Target AUC | Computational Cost | Interpretability |
|-------|-----------|------------|-------------------|------------------|
| **Core5 Baseline** | 60-65% | 0.65-0.70 | Low | High |
| **EEGNet** | 65-70% | 0.70-0.75 | Low | Medium |
| **Temporal CNN** | 70-75% | 0.75-0.80 | Medium | Medium |
| **BiLSTM** | 70-75% | 0.75-0.80 | High | Low |
| **Hybrid CNN-RNN** | 75-80% | 0.80-0.85 | High | Low |

### Success Criteria per Architecture

**Gate 1: EEGNet (Baseline DL)**
- [ ] ≥65% LOSO BA (minimum 5% improvement over Core5)
- [ ] Statistical significance (p < 0.05 vs Core5)
- [ ] Stable training across all sites
- [ ] <100ms inference time

**Gate 2: Specialized Architectures**
- [ ] ≥70% LOSO BA (≥10% improvement over Core5)
- [ ] Consistent performance across all test sites
- [ ] Interpretable attention patterns
- [ ] Clinically relevant feature learning

**Gate 3: Advanced Hybrid Model**
- [ ] ≥75% LOSO BA (≥15% improvement over Core5)
- [ ] Robust uncertainty estimation
- [ ] Deployment feasibility analysis
- [ ] Clinical validation pathway defined

---

## 🔍 **INTERPRETABILITY & EXPLAINABILITY**

### Attention Visualization
```bash
# Generate attention maps for clinical interpretation
python scripts/phase5_attention_analysis.py \
  --model temporal_cnn \
  --output_dir results/phase5/attention_maps/

# Outputs:
# - Temporal attention: Which time periods matter?
# - Spatial attention: Which brain regions contribute?
# - Clinical correlation: How do patterns relate to β-bursts?
```

### SHAP Analysis
```bash
# Generate SHAP explanations for model decisions
python scripts/phase5_shap_analysis.py \
  --model hybrid \
  --n_samples 100 \
  --output_dir results/phase5/shap_analysis/

# Clinical insights:
# - Feature importance rankings
# - Individual prediction explanations
# - Comparison with Core5 biomarker patterns
```

### Clinical Validation of Learned Features
```bash
# Compare learned features with known physiology
python scripts/phase5_clinical_validation.py \
  --learned_features results/phase5/attention_maps/ \
  --core5_biomarkers results/phase2_featopt/ablation_results.json

# Analysis:
# - Do attention patterns align with motor/posterior β-bursts?
# - Are learned temporal patterns consistent with burst physiology?
# - Can deep learning features be mapped to clinical concepts?
```

---

## 🎯 **TROUBLESHOOTING & OPTIMIZATION**

### Common Issues & Solutions

**Issue: "CUDA out of memory"**
```bash
# Solutions:
# 1. Reduce batch size
sed -i 's/batch_size: 32/batch_size: 16/' config/phase5_models.yaml

# 2. Enable gradient accumulation
sed -i 's/accumulate_grad_batches: 1/accumulate_grad_batches: 2/' config/phase5_models.yaml

# 3. Use mixed precision training (already enabled in config)
```

**Issue: "Poor convergence / NaN losses"**
```bash
# Solutions:
# 1. Lower learning rate
python scripts/phase5_train_*.py --learning_rate 1e-4

# 2. Increase gradient clipping
# 3. Check data normalization
python scripts/phase5_debug_training.py --model eegnet
```

**Issue: "Overfitting to training sites"**
```bash
# Solutions:
# 1. Increase regularization
# 2. Use more aggressive data augmentation
# 3. Reduce model complexity
python scripts/phase5_regularization_sweep.py --model temporal_cnn
```

**Issue: "Performance below Core5 baseline"**
```bash
# Diagnostic steps:
# 1. Verify data preprocessing pipeline
python scripts/phase5_data_verification.py

# 2. Check for data leakage between train/test
python scripts/phase5_leakage_check.py

# 3. Compare feature representations
python scripts/phase5_feature_analysis.py
```

---

## 📈 **RESULTS ANALYSIS & REPORTING**

### Model Comparison Dashboard
```bash
# Generate comprehensive comparison report
python scripts/phase5_generate_report.py \
  --results_dir results/phase5/ \
  --output results/phase5/phase5_final_report.html

# Interactive dashboard with:
# - Performance metrics across all models
# - LOSO validation results
# - Statistical significance tests
# - Interpretability analysis
# - Clinical insights
```

### Publication-Ready Figures
```bash
# Generate publication-quality figures
python scripts/phase5_create_figures.py \
  --results results/phase5/loso_results.json \
  --output results/phase5/figures/

# Figures generated:
# - Model comparison bar plots
# - ROC curves per site and model
# - Attention map visualizations
# - Performance trajectory (Phase II → Phase V)
# - Statistical significance matrices
```

---

## 🚀 **SUCCESS SCENARIOS & NEXT STEPS**

### Scenario 1: Breakthrough Performance (≥75% BA)
**Next Steps:**
1. **Immediate**: Document results, prepare manuscript
2. **Short-term**: Clinical validation with expert review
3. **Medium-term**: Phase VI prospective AI clinical trial
4. **Long-term**: AI/ML regulatory submission pathway

### Scenario 2: Solid Improvement (65-75% BA)
**Next Steps:**
1. **Optimization**: Advanced ensemble methods, architecture search
2. **Validation**: Extended to additional public datasets
3. **Translation**: Hybrid classical+DL deployment strategy
4. **Research**: Platform extension to other neurological conditions

### Scenario 3: Modest Gains (60-70% BA)
**Analysis Required:**
1. **Technical**: Architecture limitations, data quality issues
2. **Methodological**: LOSO validation artifacts, site-specific effects
3. **Strategic**: Focus on interpretability and clinical adoption
4. **Timeline**: Consider classical ML optimization before DL expansion

---

## 🎯 **REGULATORY & CLINICAL POSITIONING**

### Phase V as R&D Foundation
- **Current Status**: Exploratory development, not regulatory submission
- **Comparison Framework**: Deep learning as enhancement to validated Core5
- **Risk Management**: Maintain Core5 as interpretable, approved baseline
- **Future Pathway**: AI/ML evidence for v2.0 regulatory submission

### Clinical Integration Strategy
- **Immediate**: Demonstrate improved accuracy with maintained interpretability
- **Short-term**: Clinical expert validation of learned patterns
- **Medium-term**: Pilot deployment in research clinical settings
- **Long-term**: Full clinical integration with regulatory approval

---

## 📞 **PHASE V COMPLETION CRITERIA**

### Technical Deliverables
- [ ] All architectures implemented and validated
- [ ] Complete LOSO validation across 3+ sites
- [ ] Statistical comparison with Core5 baseline
- [ ] Interpretability analysis with clinical insights
- [ ] Deployment feasibility assessment

### Documentation Deliverables
- [ ] Phase V results report (comprehensive performance analysis)
- [ ] Model comparison study (statistical validation)
- [ ] Clinical interpretation guide (explainability for medical professionals)
- [ ] Technical implementation guide (reproducible methods)

### Strategic Deliverables
- [ ] Performance assessment vs Phase V targets
- [ ] Recommendation for Phase VI (advanced AI clinical trial)
- [ ] Regulatory pathway definition for AI/ML enhancement
- [ ] Platform extension opportunities (other neurological conditions)

---

**Phase V represents the frontier of AI-powered neurological assessment - building upon the solid foundation of validated Core5 biomarkers to explore the full potential of deep learning for clinical decision support.**

**Execute systematically. Validate rigorously. Position strategically. Transform EEG biomarkers from classical ML to next-generation AI.** 🚀