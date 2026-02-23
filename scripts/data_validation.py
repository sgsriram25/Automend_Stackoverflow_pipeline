"""
Data Validation Module
=======================
Validates data quality, generates schema and statistics using
Great Expectations and custom validation rules.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter

import pandas as pd
import numpy as np

from config import config

# Configure logging with UTF-8 encoding for Windows compatibility
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'validation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Schema Definition
# =============================================================================

EXPECTED_SCHEMA = {
    "required_columns": [
        "question_id", "title", "question_body", "answer_body",
        "tags", "score", "quality_score"
    ],
    "column_types": {
        "question_id": "int",
        "title": "str",
        "question_body": "str",
        "answer_body": "str",
        "tags": "list",
        "score": "int",
        "answer_score": "int",
        "view_count": "int",
        "quality_score": "float",
        "error_signatures": "list",
        "infra_components": "list",
        "question_type": "str",
        "complexity": "str",
    },
    "constraints": {
        "question_id": {"min": 1},
        "score": {"min": 0},
        "quality_score": {"min": 0},
        "question_body": {"min_length": 10},
        "answer_body": {"min_length": 20},
    }
}


# =============================================================================
# Validation Functions
# =============================================================================

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        self.statistics = {}
    
    def add_pass(self, check_name: str, message: str):
        self.passed.append({"check": check_name, "message": message})
        logger.info(f"[PASS] {check_name} - {message}")
    
    def add_failure(self, check_name: str, message: str, severity: str = "error"):
        self.failed.append({"check": check_name, "message": message, "severity": severity})
        logger.error(f"[FAIL] {check_name} - {message}")
    
    def add_warning(self, check_name: str, message: str):
        self.warnings.append({"check": check_name, "message": message})
        logger.warning(f"[WARN] {check_name} - {message}")
    
    @property
    def is_valid(self) -> bool:
        """Returns True if no critical failures."""
        critical_failures = [f for f in self.failed if f.get("severity") == "error"]
        return len(critical_failures) == 0
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "total_checks": len(self.passed) + len(self.failed),
            "passed_count": len(self.passed),
            "failed_count": len(self.failed),
            "warning_count": len(self.warnings),
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "statistics": self.statistics,
        }


def validate_schema(df: pd.DataFrame, result: ValidationResult) -> None:
    """Validate DataFrame schema against expected schema."""
    
    # Check required columns
    missing_cols = set(EXPECTED_SCHEMA["required_columns"]) - set(df.columns)
    if missing_cols:
        result.add_failure("schema_required_columns", 
                          f"Missing required columns: {missing_cols}")
    else:
        result.add_pass("schema_required_columns", "All required columns present")
    
    # Check for extra unexpected columns (warning only)
    expected_cols = set(EXPECTED_SCHEMA["column_types"].keys())
    extra_cols = set(df.columns) - expected_cols - {"processed_at", "question_metrics", "answer_metrics"}
    if extra_cols:
        result.add_warning("schema_extra_columns", f"Unexpected columns found: {extra_cols}")


def validate_data_quality(df: pd.DataFrame, result: ValidationResult) -> None:
    """Validate data quality metrics."""
    
    total_rows = len(df)
    result.statistics["total_rows"] = total_rows
    
    # Check minimum rows
    if total_rows < config.MIN_ROWS:
        result.add_failure("min_rows", 
                          f"Only {total_rows} rows, minimum required: {config.MIN_ROWS}")
    else:
        result.add_pass("min_rows", f"Row count ({total_rows}) meets minimum ({config.MIN_ROWS})")
    
    # Check for missing values in critical columns
    critical_cols = ["question_id", "question_body", "answer_body"]
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            missing_ratio = missing / total_rows
            result.statistics[f"{col}_missing_ratio"] = missing_ratio
            
            if missing_ratio > config.MAX_MISSING_RATIO:
                result.add_failure("missing_values", 
                                  f"Column '{col}' has {missing_ratio:.1%} missing values")
            elif missing > 0:
                result.add_warning("missing_values",
                                  f"Column '{col}' has {missing} missing values ({missing_ratio:.1%})")
            else:
                result.add_pass("missing_values", f"Column '{col}' has no missing values")
    
    # Check for duplicates
    if "question_id" in df.columns:
        duplicates = df.duplicated(subset=["question_id"]).sum()
        dup_ratio = duplicates / total_rows
        result.statistics["duplicate_ratio"] = dup_ratio
        
        if dup_ratio > config.MAX_DUPLICATE_RATIO:
            result.add_failure("duplicates", 
                              f"Found {duplicates} duplicate question_ids ({dup_ratio:.1%})")
        elif duplicates > 0:
            result.add_warning("duplicates", f"Found {duplicates} duplicates")
        else:
            result.add_pass("duplicates", "No duplicate question_ids found")
    
    # Check for empty strings
    text_cols = ["question_body", "answer_body", "title"]
    for col in text_cols:
        if col in df.columns:
            empty = (df[col].str.strip() == "").sum()
            if empty > 0:
                result.add_warning("empty_strings", f"Column '{col}' has {empty} empty strings")


def validate_text_quality(df: pd.DataFrame, result: ValidationResult) -> None:
    """Validate text content quality."""
    
    # Question body length
    if "question_body" in df.columns:
        q_lengths = df["question_body"].str.len()
        result.statistics["question_length"] = {
            "min": int(q_lengths.min()),
            "max": int(q_lengths.max()),
            "mean": float(q_lengths.mean()),
            "median": float(q_lengths.median())
        }
        
        short_questions = (q_lengths < 50).sum()
        if short_questions > len(df) * 0.1:
            result.add_warning("text_quality", 
                              f"{short_questions} questions are very short (<50 chars)")
        else:
            result.add_pass("text_quality", "Question lengths are acceptable")
    
    # Answer body length
    if "answer_body" in df.columns:
        a_lengths = df["answer_body"].str.len()
        result.statistics["answer_length"] = {
            "min": int(a_lengths.min()),
            "max": int(a_lengths.max()),
            "mean": float(a_lengths.mean()),
            "median": float(a_lengths.median())
        }
        
        short_answers = (a_lengths < 100).sum()
        if short_answers > len(df) * 0.1:
            result.add_warning("text_quality", 
                              f"{short_answers} answers are very short (<100 chars)")


def validate_feature_distribution(df: pd.DataFrame, result: ValidationResult) -> None:
    """Validate feature distributions for potential issues."""
    
    # Score distribution
    if "score" in df.columns:
        score_stats = df["score"].describe()
        result.statistics["score_distribution"] = score_stats.to_dict()
        
        # Check for anomalies
        if score_stats["std"] == 0:
            result.add_warning("feature_distribution", "All scores are identical")
    
    # Quality score distribution
    if "quality_score" in df.columns:
        qs_stats = df["quality_score"].describe()
        result.statistics["quality_score_distribution"] = qs_stats.to_dict()
    
    # Tag distribution
    if "tags" in df.columns:
        all_tags = []
        for tags in df["tags"]:
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        tag_counts = Counter(all_tags)
        result.statistics["tag_distribution"] = dict(tag_counts.most_common(20))
        
        if len(tag_counts) < 3:
            result.add_warning("feature_distribution", "Very few unique tags found")
    
    # Error signature distribution
    if "error_signatures" in df.columns:
        all_errors = []
        for errors in df["error_signatures"]:
            if isinstance(errors, list):
                all_errors.extend(errors)
        
        error_counts = Counter(all_errors)
        result.statistics["error_signature_distribution"] = dict(error_counts)


def validate_anomalies(df: pd.DataFrame, result: ValidationResult) -> list:
    """Detect anomalies in the data. Returns list of anomalous records."""
    
    anomalies = []
    
    # Detect outliers in numeric columns using IQR
    numeric_cols = ["score", "answer_score", "view_count", "quality_score"]
    
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                result.statistics[f"{col}_outliers"] = len(outliers)
                
                # Only warn if significant number of outliers
                if len(outliers) > len(df) * 0.05:
                    result.add_warning("anomaly_detection",
                                      f"Found {len(outliers)} outliers in '{col}'")
                
                for idx, row in outliers.iterrows():
                    anomalies.append({
                        "type": "outlier",
                        "column": col,
                        "question_id": row.get("question_id"),
                        "value": row[col],
                        "bounds": [lower_bound, upper_bound]
                    })
    
    # Detect suspicious text patterns
    if "question_body" in df.columns:
        # Check for HTML that wasn't cleaned
        html_pattern = df["question_body"].str.contains(r'<[a-z]+>', regex=True, na=False)
        html_count = html_pattern.sum()
        if html_count > 0:
            result.add_warning("anomaly_detection", 
                              f"Found {html_count} records with uncleaned HTML")
    
    result.statistics["total_anomalies"] = len(anomalies)
    return anomalies


# =============================================================================
# Statistics Generation
# =============================================================================

def generate_statistics(df: pd.DataFrame) -> dict:
    """Generate comprehensive statistics for the dataset."""
    
    stats = {
        "generated_at": datetime.now().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }
    
    # Numeric column statistics
    numeric_stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        numeric_stats[col] = {
            "count": int(df[col].count()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "25%": float(df[col].quantile(0.25)),
            "50%": float(df[col].quantile(0.50)),
            "75%": float(df[col].quantile(0.75)),
            "max": float(df[col].max()),
        }
    stats["numeric_statistics"] = numeric_stats
    
    # Text column statistics
    text_cols = ["title", "question_body", "answer_body"]
    text_stats = {}
    for col in text_cols:
        if col in df.columns:
            lengths = df[col].str.len()
            text_stats[col] = {
                "avg_length": float(lengths.mean()),
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "total_chars": int(lengths.sum()),
            }
    stats["text_statistics"] = text_stats
    
    # Categorical statistics
    if "question_type" in df.columns:
        stats["question_type_distribution"] = df["question_type"].value_counts().to_dict()
    
    if "complexity" in df.columns:
        stats["complexity_distribution"] = df["complexity"].value_counts().to_dict()
    
    return stats


# =============================================================================
# Main Validation Pipeline
# =============================================================================

def run_validation() -> dict:
    """
    Main validation function - entry point for Airflow task.
    
    Returns:
        dict: Validation results and statistics
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA VALIDATION")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    result = ValidationResult()
    
    try:
        # Load processed data
        input_path = config.processed_dir / "qa_pairs_processed.json"
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records for validation")
        
        # Run validations
        validate_schema(df, result)
        validate_data_quality(df, result)
        validate_text_quality(df, result)
        validate_feature_distribution(df, result)
        
        # Detect anomalies
        anomalies = validate_anomalies(df, result)
        
        # Generate statistics
        statistics = generate_statistics(df)
        result.statistics.update(statistics)
        
        # Save results
        validation_output = result.to_dict()
        validation_output["start_time"] = start_time.isoformat()
        validation_output["end_time"] = datetime.now().isoformat()
        
        # Save validation report
        report_path = config.REPORTS_DIR / "validation" / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(validation_output, f, indent=2, default=str)
        
        # Save statistics separately
        stats_path = config.REPORTS_DIR / "statistics" / "data_statistics.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(statistics, f, indent=2, default=str)
        
        # Save metrics file for DVC tracking
        metrics = {
            "is_valid": result.is_valid,
            "passed_checks": len(result.passed),
            "failed_checks": len(result.failed),
            "warning_checks": len(result.warnings),
            "total_records": len(df),
            "anomaly_count": result.statistics.get("total_anomalies", 0),
            "duplicate_count": result.statistics.get("duplicate_ratio", 0) * len(df),
            "timestamp": datetime.now().isoformat()
        }
        
        metrics_path = config.REPORTS_DIR / "validation" / "validation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save anomalies
        if anomalies:
            anomaly_path = config.REPORTS_DIR / "validation" / "anomalies.json"
            with open(anomaly_path, "w") as f:
                json.dump(anomalies, f, indent=2, default=str)
        
        # If valid, copy to validated directory
        if result.is_valid:
            validated_path = config.validated_dir / "qa_pairs_validated.json"
            with open(validated_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Data validated and saved to {validated_path}")
        else:
            logger.error("Validation failed - data not promoted to validated directory")
        
        logger.info(f"Validation complete: {len(result.passed)} passed, "
                   f"{len(result.failed)} failed, {len(result.warnings)} warnings")
        
        return validation_output
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        result.add_failure("execution", str(e))
        raise


if __name__ == "__main__":
    run_validation()