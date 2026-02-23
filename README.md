# AutoMend Data Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Airflow 2.9](https://img.shields.io/badge/airflow-2.9-green.svg)](https://airflow.apache.org/)
[![DVC](https://img.shields.io/badge/dvc-3.30+-purple.svg)](https://dvc.org/)

A production-ready MLOps data pipeline for processing StackOverflow Q&A data. Built with Apache Airflow for orchestration, DVC for data versioning, and comprehensive validation and bias detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Data Versioning with DVC](#data-versioning-with-dvc)
- [Pipeline Components](#pipeline-components)
- [Testing](#testing)
- [Monitoring & Alerts](#monitoring--alerts)
- [Bias Detection & Mitigation](#bias-detection--mitigation)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

AutoMend transforms StackOverflow troubleshooting data into structured training datasets for ML-powered infrastructure remediation systems.

### Key Features

- ✅ **Data Acquisition** - Fetch from StackOverflow API or CSV files
- ✅ **Data Preprocessing** - Cleaning, feature extraction, quality scoring
- ✅ **Data Validation** - Schema validation, statistics generation, anomaly detection
- ✅ **Bias Detection** - Data slicing and fairness analysis
- ✅ **Bias Mitigation** - Resampling and quality weighting
- ✅ **Pipeline Orchestration** - Airflow DAGs with error handling
- ✅ **Data Versioning** - DVC for reproducibility
- ✅ **Alerting** - Slack/Email notifications on failures
- ✅ **Comprehensive Testing** - pytest with 40+ unit tests

## 📁 Project Structure

```
AutoMend-Pipeline/
├── dags/
│   └── automend_pipeline_dag.py    # Airflow DAG definition
│
├── scripts/
│   ├── __init__.py
│   ├── config.py                   # Central configuration
│   ├── data_acquisition.py         # Data fetching (API/CSV)
│   ├── data_preprocessing.py       # Data cleaning & features
│   ├── data_validation.py          # Validation & anomaly detection
│   ├── schema_validation.py        # Great Expectations integration
│   ├── bias_detection.py           # Bias analysis
│   ├── generate_training_data.py   # Training data with mitigation
│   └── alerts.py                   # Slack/Email notifications
│
├── tests/
│   ├── __init__.py
│   └── test_modules.py             # Unit tests (pytest)
│
├── data/
│   ├── external/                   # Source CSV files
│   ├── raw/                        # Raw fetched data
│   ├── processed/                  # Cleaned data
│   ├── validated/                  # Validated data
│   └── training/                   # Final training data
│
├── logs/                           # Pipeline logs
├── reports/
│   ├── validation/                 # Validation reports
│   ├── statistics/                 # Data statistics
│   └── bias/                       # Bias analysis reports
│
├── docker-compose.yaml             # Airflow Docker setup
├── dvc.yaml                        # DVC pipeline definition
├── params.yaml                     # Pipeline parameters
├── pyproject.toml                  # Dependencies (UV)
├── requirements.txt                # Dependencies (pip)
├── .env                            # Environment variables
└── README.md
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for Airflow)
- Git

### Option 1: Using UV (Recommended - Fastest)

```bash
# Install UV
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <your-repo-url>
cd AutoMend-Pipeline

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Initialize DVC
dvc init
```

### Option 2: Using pip

```bash
git clone <your-repo-url>
cd AutoMend-Pipeline

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

dvc init
```

### Environment Setup

```bash
# Create .env file
cat > .env << EOF
AIRFLOW_UID=50000

# Optional: Slack notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Optional: Email notifications
ALERT_EMAIL=your-email@example.com

# Optional: StackOverflow API key (for higher rate limits)
STACKOVERFLOW_API_KEY=your_api_key
EOF
```

### Data Setup

Copy your CSV files to the data directory:

```bash
mkdir -p data/external
cp /path/to/Stack_Qns_pl.csv data/external/
cp /path/to/Stack_Ans_pl.csv data/external/
```

## ▶️ Running the Pipeline

### Method 1: Run Scripts Directly (Development)

```bash
# Activate environment
source .venv/bin/activate

# Run each stage
python scripts/data_acquisition.py --use-csv
python scripts/data_preprocessing.py
python scripts/data_validation.py
python scripts/schema_validation.py
python scripts/bias_detection.py
python scripts/generate_training_data.py

# Run tests
pytest tests/ -v
```

### Method 2: Using DVC (Reproducible Pipeline)

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro validate

# View pipeline DAG
dvc dag
```

### Method 3: Using Airflow (Production)

```bash
# Start Airflow
docker-compose up -d

# Wait for initialization (~2-3 minutes)
docker-compose logs -f airflow-init

# Access Airflow UI
# http://localhost:8080
# Username: airflow
# Password: airflow

# Enable and trigger the DAG
# Or via CLI:
docker-compose exec airflow-worker airflow dags trigger automend_data_pipeline
```

## 📊 Data Versioning with DVC

### Initial Setup

```bash
# Initialize DVC (if not done)
dvc init

# Add remote storage (optional but recommended)
dvc remote add -d myremote s3://your-bucket/dvc-store
# Or use local storage:
dvc remote add -d myremote /path/to/dvc-storage

# Track data files
dvc add data/external/Stack_Qns_pl.csv
dvc add data/external/Stack_Ans_pl.csv

# Commit DVC files to Git
git add data/external/*.dvc data/external/.gitignore
git commit -m "Track source data with DVC"
```

### Reproducing the Pipeline

```bash
# On a new machine, clone the repo
git clone <your-repo>
cd AutoMend-Pipeline

# Pull data from DVC remote
dvc pull

# Reproduce the pipeline
dvc repro
```

### Versioning Data Changes

```bash
# After data changes, update DVC tracking
dvc add data/raw/qa_pairs_raw.json

# Push new data version
dvc push

# Commit changes
git add data/raw/qa_pairs_raw.json.dvc
git commit -m "Update raw data"
git push
```

### Viewing Pipeline

```bash
# View DAG
dvc dag

# View metrics
dvc metrics show

# Compare with previous version
dvc metrics diff
```

## 🔧 Pipeline Components

### 1. Data Acquisition (`data_acquisition.py`)
- Fetches Q&A data from StackOverflow API or local CSV
- Rate limiting with exponential backoff
- Batch fetching (100 answers per request)

### 2. Data Preprocessing (`data_preprocessing.py`)
- HTML cleaning and text normalization
- Feature extraction: error signatures, infrastructure components
- Quality score calculation
- Question type classification

### 3. Data Validation (`data_validation.py`)
- Schema validation against expected structure
- Data quality checks (missing values, duplicates)
- Anomaly detection using IQR method

### 4. Schema Validation (`schema_validation.py`)
- Great Expectations integration (optional)
- Automated statistics generation
- Data profiling and documentation

### 5. Bias Detection (`bias_detection.py`)
- Data slicing by tags, error types, complexity
- Bias metrics calculation
- Mitigation suggestions

### 6. Training Data Generation (`generate_training_data.py`)
- Bias mitigation via resampling
- Quality-based sample weighting
- Train/test split
- Multiple output formats (JSON, JSONL)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=scripts --cov-report=html

# Run specific test class
pytest tests/test_modules.py::TestDataPreprocessing -v

# Run with verbose output
pytest tests/ -v --tb=long
```

### Test Coverage

| Module | Tests |
|--------|-------|
| Data Acquisition | 7 tests |
| Data Preprocessing | 12 tests |
| Data Validation | 8 tests |
| Bias Detection | 5 tests |
| Integration | 3 tests |
| Edge Cases | 5 tests |

## 📈 Monitoring & Alerts

### Airflow Monitoring

1. **Gantt Chart**: View task execution timeline
   - Access: Airflow UI → DAG → Gantt view
   - Identify bottlenecks and optimize

2. **Task Logs**: Debug failures
   - Click on failed task → Log

### Alert Channels

Configure in `.env`:

```bash
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Email
ALERT_EMAIL=team@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
```

### Alerts Triggered For:
- Pipeline start/completion
- Validation failures
- Anomaly detection (outliers)
- Bias detection
- Task failures

## ⚖️ Bias Detection & Mitigation

### Detected Bias Types

1. **Underrepresentation**: Slices with <1% of data
2. **Quality Disparity**: >25% difference in quality scores
3. **Content Length Bias**: >50% difference in answer lengths

### Mitigation Strategies

1. **Resampling**: Oversample underrepresented slices
2. **Quality Weighting**: Assign sample weights based on quality
3. **Stratified Splitting**: Maintain distribution in train/test

### Viewing Bias Report

```bash
# After running bias detection
cat reports/bias/bias_report.json | python -m json.tool

# Key sections:
# - biased_slices: List of slices with detected bias
# - mitigation_suggestions: Recommended actions
# - summary: High/medium/low severity counts
```

## 🔍 Troubleshooting

### Unicode Errors on Windows

```powershell
# Set PowerShell to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
```

### API Rate Limiting

```bash
# Use CSV mode instead of API
python scripts/data_acquisition.py --use-csv

# Or get an API key from https://stackapps.com/apps/oauth/register
```

### Docker Issues

```bash
# Reset Airflow
docker-compose down -v
docker-compose up -d

# Check logs
docker-compose logs airflow-worker --tail=100
```

### DVC Issues

```bash
# Check DVC status
dvc status

# Force reproduce
dvc repro --force

# Check remote connection
dvc remote list
```

## 📄 License

MIT License

## 👥 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests: `pytest tests/ -v`
5. Run linting: `ruff check scripts/`
6. Submit pull request

## 📚 References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [Great Expectations](https://greatexpectations.io/expectations/)
- [Fairlearn](https://fairlearn.org/)