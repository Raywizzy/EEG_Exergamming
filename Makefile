# EEG Biomarker Validation Pipeline
# Phase IV Multi-Site Clinical Validation

# Environment Management
ENV_NAME = eeg-phd
PYTHON = conda run -n $(ENV_NAME) python
CONFIG = config/preprocessing_config.yaml

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "EEG Biomarker Validation Pipeline"
	@echo "=================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          - Create conda environment and install dependencies"
	@echo "  clean-env      - Remove conda environment"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  bidsify        - Convert raw datasets to BIDS format"
	@echo "  preprocess     - Harmonized preprocessing across all sites"
	@echo "  features       - Extract Core5 biomarker features"
	@echo "  loso           - Run Leave-One-Site-Out validation"
	@echo ""
	@echo "Quality Control:"
	@echo "  qc             - Generate quality control reports"
	@echo "  dashboard      - Create interactive QC dashboard"
	@echo "  drift-check    - Monitor site performance drift"
	@echo ""
	@echo "Analysis:"
	@echo "  stats          - Statistical analysis and reporting"
	@echo "  figures        - Generate publication-quality figures"
	@echo "  report         - Complete analysis report"
	@echo ""
	@echo "Utilities:"
	@echo "  validate       - Validate pipeline integrity"
	@echo "  clean          - Clean intermediate files"
	@echo "  test           - Run unit tests"

# Environment Setup
setup:
	@echo "Setting up conda environment..."
	conda env create -f env/environment.yml -n $(ENV_NAME) || conda env update -f env/environment.yml -n $(ENV_NAME)
	@echo "Environment '$(ENV_NAME)' ready!"
	@echo "Activate with: conda activate $(ENV_NAME)"

clean-env:
	@echo "Removing conda environment..."
	conda env remove -n $(ENV_NAME) -y

# Data Pipeline Targets
bidsify:
	@echo "Converting raw datasets to BIDS format..."
	$(PYTHON) -m src.io.convert_to_bids --config $(CONFIG)
	@echo "BIDS conversion complete!"

preprocess:
	@echo "Running harmonized preprocessing..."
	$(PYTHON) -m src.preprocess.pipeline --config $(CONFIG)
	@echo "Preprocessing complete!"

features:
	@echo "Extracting Core5 biomarker features..."
	$(PYTHON) -m src.features.core5 --config $(CONFIG)
	@echo "Feature extraction complete!"

loso:
	@echo "Running Leave-One-Site-Out validation..."
	$(PYTHON) -m src.model.loso --config $(CONFIG)
	@echo "LOSO validation complete!"

# Quality Control Targets
qc:
	@echo "Generating quality control reports..."
	$(PYTHON) -m src.qc.metrics --config $(CONFIG)
	@echo "QC reports generated!"

dashboard:
	@echo "Creating interactive QC dashboard..."
	$(PYTHON) -m src.qc.dashboard --config $(CONFIG)
	@echo "Dashboard created! Open: runs/reports/qc_dashboard.html"

drift-check:
	@echo "Checking for site performance drift..."
	$(PYTHON) -m src.qc.drift_monitor --config $(CONFIG)
	@echo "Drift monitoring complete!"

# Analysis Targets
stats:
	@echo "Running statistical analysis..."
	$(PYTHON) -m src.analysis.statistics --config $(CONFIG)
	@echo "Statistical analysis complete!"

figures:
	@echo "Generating publication figures..."
	$(PYTHON) -m src.analysis.figures --config $(CONFIG)
	@echo "Figures generated in: runs/figures/"

report:
	@echo "Generating comprehensive analysis report..."
	$(PYTHON) -m src.analysis.report --config $(CONFIG)
	@echo "Report generated: runs/reports/analysis_report.html"

# Utility Targets
validate:
	@echo "Validating pipeline integrity..."
	$(PYTHON) -m src.utils.validate --config $(CONFIG)
	@echo "Pipeline validation complete!"

clean:
	@echo "Cleaning intermediate files..."
	rm -rf data/interim/*
	rm -rf runs/logs/*
	rm -rf runs/temp/*
	@echo "Cleanup complete!"

test:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/ -v
	@echo "Tests complete!"

# Regulatory Compliance Targets
audit:
	@echo "Generating audit trail..."
	$(PYTHON) -m src.compliance.audit --config $(CONFIG)
	@echo "Audit trail generated!"

regulatory-package:
	@echo "Creating regulatory submission package..."
	$(PYTHON) -m src.compliance.package --config $(CONFIG)
	@echo "Regulatory package ready!"

# Full Pipeline Execution
all: bidsify preprocess features loso qc stats report
	@echo "Complete pipeline execution finished!"

# Phase-Specific Targets
phase3-reproduce:
	@echo "Reproducing Phase III results..."
	$(PYTHON) -m src.model.phase3_reproduce --config $(CONFIG)
	@echo "Phase III reproduction complete!"

phase4-validate:
	@echo "Running Phase IV validation..."
	$(PYTHON) -m src.model.phase4_validate --config $(CONFIG)
	@echo "Phase IV validation complete!"

# Development Targets
dev-setup: setup
	@echo "Setting up development environment..."
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install pytest black flake8 mypy
	@echo "Development environment ready!"

format:
	@echo "Formatting code..."
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m flake8 src/ tests/
	@echo "Code formatting complete!"

# Quick Start Target
quickstart: setup bidsify preprocess
	@echo "Quick start setup complete!"
	@echo "Next steps:"
	@echo "  1. make features"
	@echo "  2. make loso"
	@echo "  3. make qc"

# Monitoring Targets
monitor-weekly:
	@echo "Running weekly monitoring checks..."
	$(PYTHON) -m src.monitoring.weekly --config $(CONFIG)
	@echo "Weekly monitoring complete!"

monitor-realtime:
	@echo "Starting real-time monitoring..."
	$(PYTHON) -m src.monitoring.realtime --config $(CONFIG)

# Documentation Targets
docs:
	@echo "Generating documentation..."
	$(PYTHON) -m sphinx-build -b html docs/ docs/_build/html
	@echo "Documentation generated: docs/_build/html/index.html"

# Phony targets
.PHONY: help setup clean-env bidsify preprocess features loso qc dashboard drift-check stats figures report validate clean test audit regulatory-package all phase3-reproduce phase4-validate dev-setup format quickstart monitor-weekly monitor-realtime docs