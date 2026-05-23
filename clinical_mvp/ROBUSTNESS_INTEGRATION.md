# Core15+ Robustness Testing Integration

## Overview

The Core15+ Clinical MVP now includes comprehensive robustness testing capabilities that systematically evaluate pipeline performance under controlled degradations. This integration provides clinicians with confidence metrics about the biomarker's performance across varying conditions.

## 🧪 Robustness Testing Framework

### Test Factors
- **Noise Levels**: none, low, medium, high
- **Channel Dropouts**: 0, 5, 10, 20 channels
- **Duration Limits**: full length, 120s, 60s
- **Sampling Rates**: 256Hz, 128Hz, 64Hz

### Performance Benchmarks
- **Baseline Target**: 97.2% balanced accuracy
- **Clinical Threshold**: 65% balanced accuracy
- **Excellent Performance**: ≥90%
- **Good Performance**: 75-89%
- **Acceptable Performance**: 65-74%

## 🔗 API Integration

### New Endpoints

#### `POST /api/robustness`
Start comprehensive robustness testing suite.

```json
{
  "bids_dir": "/path/to/bids/data",
  "site_id": "ROBUSTNESS",
  "trial_id": "VALIDATION"
}
```

Response:
```json
{
  "job_id": "uuid",
  "type": "robustness",
  "status": "started",
  "message": "Robustness testing initiated..."
}
```

#### `GET /api/robustness/results`
Get latest robustness testing results with summary statistics.

```json
{
  "summary": {
    "total_runs": 64,
    "successful_runs": 62,
    "mean_ba": 89.3,
    "std_ba": 8.2,
    "min_ba": 71.5,
    "max_ba": 97.8,
    "below_threshold": 2
  },
  "NOISE_LEVEL": {
    "none": {"mean_ba": 97.2, "count": 16},
    "low": {"mean_ba": 94.1, "count": 16},
    "medium": {"mean_ba": 88.7, "count": 16},
    "high": {"mean_ba": 82.4, "count": 16}
  },
  "files": {
    "summary_csv": "/path/to/robustness_summary.csv",
    "plots_directory": "/path/to/plots"
  }
}
```

#### `GET /api/robustness/status`
Get status of latest robustness testing job.

```json
{
  "job_id": "uuid",
  "status": "succeeded",
  "created_at": "2025-01-20T10:30:00",
  "runtime_sec": 1847.2,
  "message": "Latest robustness test is succeeded"
}
```

## 🖥️ Web Interface Integration

### Jobs Page Enhancement
- **"Run Robustness Test" Button**: One-click robustness testing
- **Robustness Results Indicator**: Shows when results are available
- **Job Type Identification**: Distinguishes robustness jobs from validation jobs

### User Experience
1. Click "Run Robustness Test" on Jobs page
2. Enter BIDS dataset path
3. Monitor progress in real-time (2+ hour runtime expected)
4. View comprehensive results and plots
5. Download CSV data and visualization plots

## 📊 Results and Visualization

### Generated Outputs
- **`robustness_summary.csv`**: Complete results for all test combinations
- **`robustness_panel.json`**: Summary metrics for web display
- **`plots/robustness_overview.png`**: Performance by factor visualization
- **`plots/degradation_distribution.png`**: Overall performance distribution

### Plot Features
- Color-coded performance bars (Green: Excellent, Orange: Good, Red: Acceptable)
- Reference lines for baseline and clinical threshold
- Error bars showing variability
- Value labels on each bar

## 🏗️ Implementation Architecture

### File Structure
```
clinical_mvp/
├── robustness/
│   └── robustness_harness.py     # Main testing framework
├── app/
│   ├── routes/jobs.py            # API endpoints
│   ├── services/jobs.py          # Job management
│   └── templates/jobs.html       # UI integration
└── test_robustness_integration.py # Integration tests
```

### Job Management Integration
- **JobManager.create_robustness_job()**: Creates robustness testing jobs
- **Background Execution**: Non-blocking 2+ hour execution
- **Status Tracking**: Real-time progress monitoring
- **Results Loading**: Automatic metric extraction

## 🔬 Clinical Applications

### Quality Assurance
- Validate biomarker robustness before clinical deployment
- Assess performance across different recording conditions
- Identify optimal data collection parameters

### Regulatory Compliance
- Comprehensive performance characterization
- Statistical evidence for clinical claims
- Documentation for regulatory submissions

### Research Applications
- Dataset quality assessment
- Cross-site validation
- Performance benchmarking

## 🚀 Usage Instructions

### 1. Deploy Platform
```bash
cd clinical_mvp
docker-compose up -d
```

### 2. Access Web Interface
Navigate to: http://localhost:8080/ui/jobs

### 3. Start Robustness Test
- Click "Run Robustness Test"
- Enter BIDS dataset path
- Monitor progress (expect 30-120 minutes)

### 4. View Results
- Check job status in dashboard
- Click robustness results indicator when available
- Download plots and CSV data

### 5. API Usage
```bash
# Start robustness test
curl -X POST "http://localhost:8080/api/robustness" \
  -H "Content-Type: application/json" \
  -d '{"bids_dir": "/path/to/data"}'

# Check results
curl "http://localhost:8080/api/robustness/results"
```

## 📈 Performance Interpretation

### Robustness Score Interpretation
- **95%+**: Exceptional robustness across all conditions
- **90-94%**: Excellent robustness suitable for clinical deployment
- **80-89%**: Good robustness with minor performance drops
- **70-79%**: Moderate robustness requiring careful condition control
- **<70%**: Limited robustness requiring optimization

### Factor-Specific Analysis
- **Noise Tolerance**: Performance under artifact conditions
- **Channel Robustness**: Minimal electrode requirements
- **Duration Flexibility**: Short recording capability
- **Sampling Rate**: Minimum acquisition requirements

## 🔧 Technical Details

### Test Execution
- **Total Combinations**: 4 × 4 × 3 × 3 = 144 tests
- **Parallel Processing**: Configurable for multi-core systems
- **Memory Management**: Automatic cleanup between tests
- **Error Handling**: Graceful degradation on test failures

### Result Aggregation
- **Statistical Summaries**: Mean, std, min, max per factor
- **Performance Thresholds**: Clinical significance assessment
- **Visualization**: Automated plot generation
- **Export Formats**: CSV, JSON, PNG plots

## 🎯 Clinical Validation Benefits

### Confidence Building
- Quantified performance ranges under realistic conditions
- Evidence-based deployment decisions
- Risk assessment for clinical applications

### Quality Control
- Automated acceptance criteria
- Performance monitoring
- Continuous validation framework

### Regulatory Readiness
- Comprehensive performance characterization
- Statistical documentation
- Clinical trial support

---

**🧪 Robustness Testing Complete**
Integrated into Core15+ Clinical Platform v1.0 | Ready for Clinical Deployment