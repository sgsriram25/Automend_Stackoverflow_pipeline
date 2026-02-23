"""
AutoMend Pipeline Configuration
================================
Central configuration for all pipeline components.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Determine base directory (works for both local and Docker)
def get_base_dir() -> Path:
    """Get the base directory for the project."""
    # Check if running in Airflow/Docker
    if os.environ.get("AIRFLOW_HOME"):
        return Path(os.environ["AIRFLOW_HOME"])
    # Check if /opt/airflow exists (Docker)
    if Path("/opt/airflow").exists():
        return Path("/opt/airflow")
    # Default to parent of scripts directory
    return Path(__file__).parent.parent


@dataclass
class PipelineConfig:
    """Main configuration class for the AutoMend pipeline."""
    
    # === Base Paths ===
    BASE_DIR: Path = field(default_factory=get_base_dir)
    
    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / "data"
    
    @property
    def LOGS_DIR(self) -> Path:
        return self.BASE_DIR / "logs"
    
    @property
    def SCHEMAS_DIR(self) -> Path:
        return self.BASE_DIR / "schemas"
    
    @property
    def REPORTS_DIR(self) -> Path:
        return self.BASE_DIR / "reports"
    
    # === Data Paths ===
    @property
    def raw_dir(self) -> Path:
        return self.DATA_DIR / "raw"
    
    @property
    def processed_dir(self) -> Path:
        return self.DATA_DIR / "processed"
    
    @property
    def validated_dir(self) -> Path:
        return self.DATA_DIR / "validated"
    
    @property
    def training_dir(self) -> Path:
        return self.DATA_DIR / "training"
    
    # === CSV Fallback Paths ===
    @property
    def CSV_QUESTIONS_PATH(self) -> Path:
        env_path = os.getenv("CSV_QUESTIONS_PATH")
        if env_path:
            return Path(env_path)
        return self.DATA_DIR / "external" / "Stack_Qns_pl.csv"
    
    @property
    def CSV_ANSWERS_PATH(self) -> Path:
        env_path = os.getenv("CSV_ANSWERS_PATH")
        if env_path:
            return Path(env_path)
        return self.DATA_DIR / "external" / "Stack_Ans_pl.csv"
    
    # === API Configuration ===
    STACKOVERFLOW_API_KEY: Optional[str] = os.getenv("STACKOVERFLOW_API_KEY")
    TAGS: tuple = (
        "kubernetes", "terraform", "gpu", "autoscaling",
        "mlops", "tensorflow", "pytorch", "airflow",
        "docker", "prometheus", "grafana"
    )
    MIN_SCORE: int = 5
    QUESTIONS_PER_TAG: int = 15
    
    # === Rate Limiting ===
    API_DELAY: float = 2.0
    BATCH_SIZE: int = 100
    MAX_RETRIES: int = 5
    BACKOFF_MULTIPLIER: float = 2.0
    MAX_BACKOFF: float = 120.0
    
    # === Alerting ===
    SLACK_WEBHOOK_URL: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")
    SLACK_CHANNEL: str = os.getenv("SLACK_CHANNEL", "#mlops-alerts")
    ALERT_EMAIL: Optional[str] = os.getenv("ALERT_EMAIL")
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    
    # === Validation Thresholds ===
    MAX_MISSING_RATIO: float = 0.1      # Max 10% missing values
    MIN_ROWS: int = 100                  # Minimum rows required
    MAX_DUPLICATE_RATIO: float = 0.05   # Max 5% duplicates
    
    # === Bias Detection ===
    BIAS_THRESHOLD: float = 0.25        # 25% performance difference threshold (was 10% - too sensitive)
    SLICE_MIN_SAMPLES: int = 30         # Min samples per slice for analysis
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.raw_dir, self.processed_dir, self.validated_dir,
                         self.training_dir, self.LOGS_DIR, self.SCHEMAS_DIR, self.REPORTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global config instance
config = PipelineConfig()


# === Error Patterns for Classification ===
ERROR_PATTERNS = {
    "CrashLoopBackOff": r"CrashLoopBackOff",
    "OOMKilled": r"OOMKilled|Out[Oo]f[Mm]emory|oom|OOM",
    "ImagePullBackOff": r"ImagePullBackOff|ErrImagePull",
    "Timeout": r"timeout|Timeout|TIMEOUT|timed out",
    "ConnectionRefused": r"connection refused|ECONNREFUSED",
    "PermissionDenied": r"permission denied|403|401|unauthorized",
    "ResourceExhausted": r"resource exhausted|quota exceeded|limit reached",
    "GPUError": r"CUDA|cuda|GPU|gpu error|device not found",
    "NetworkError": r"network|dns|resolve|unreachable",
    "ConfigError": r"config|configuration|yaml|invalid",
}

# === Infrastructure Patterns ===
INFRA_PATTERNS = {
    "kubernetes": r"kubernetes|k8s|kubectl|pod|deployment|service|ingress|helm",
    "terraform": r"terraform|tfstate|provider|module|hcl",
    "docker": r"docker|container|dockerfile|image|compose",
    "gpu": r"gpu|cuda|nvidia|GPU|tensor|TPU",
    "database": r"database|postgres|mysql|redis|mongodb|sql",
    "cloud_aws": r"aws|ec2|s3|lambda|eks|iam",
    "cloud_gcp": r"gcp|gke|bigquery|cloud run",
    "cloud_azure": r"azure|aks|blob",
    "monitoring": r"prometheus|grafana|alertmanager|datadog|metrics",
    "cicd": r"jenkins|github actions|gitlab|circleci|pipeline",
}