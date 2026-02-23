"""
Bias Detection Module
======================
Performs data slicing and bias analysis across different subgroups.
Identifies potential biases in the dataset and suggests mitigations.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional

import pandas as pd
import numpy as np

from config import config

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'bias_detection.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Slicing Functions
# =============================================================================

def create_slices_by_column(df: pd.DataFrame, column: str, min_samples: int = None) -> dict:
    """
    Create data slices based on a categorical column.
    
    Args:
        df: DataFrame to slice
        column: Column name to slice by
        min_samples: Minimum samples required per slice
    
    Returns:
        dict: {slice_value: slice_dataframe}
    """
    min_samples = min_samples or config.SLICE_MIN_SAMPLES
    slices = {}
    
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return slices
    
    for value in df[column].unique():
        slice_df = df[df[column] == value]
        if len(slice_df) >= min_samples:
            slices[value] = slice_df
        else:
            logger.debug(f"Slice '{value}' has only {len(slice_df)} samples (min: {min_samples})")
    
    return slices


def create_slices_by_tags(df: pd.DataFrame, min_samples: int = None) -> dict:
    """Create slices based on tags (each row can have multiple tags)."""
    min_samples = min_samples or config.SLICE_MIN_SAMPLES
    slices = {}
    
    if "tags" not in df.columns:
        return slices
    
    # Get all unique tags
    all_tags = set()
    for tags in df["tags"]:
        if isinstance(tags, list):
            all_tags.update(tags)
    
    # Create slice for each tag
    for tag in all_tags:
        mask = df["tags"].apply(lambda x: tag in x if isinstance(x, list) else False)
        slice_df = df[mask]
        if len(slice_df) >= min_samples:
            slices[f"tag_{tag}"] = slice_df
    
    return slices


def create_slices_by_error_signature(df: pd.DataFrame, min_samples: int = None) -> dict:
    """Create slices based on error signatures."""
    min_samples = min_samples or config.SLICE_MIN_SAMPLES
    slices = {}
    
    if "error_signatures" not in df.columns:
        return slices
    
    # Get all unique error signatures
    all_errors = set()
    for errors in df["error_signatures"]:
        if isinstance(errors, list):
            all_errors.update(errors)
    
    # Create slice for each error type
    for error in all_errors:
        mask = df["error_signatures"].apply(lambda x: error in x if isinstance(x, list) else False)
        slice_df = df[mask]
        if len(slice_df) >= min_samples:
            slices[f"error_{error}"] = slice_df
    
    return slices


# =============================================================================
# Bias Metrics Calculation
# =============================================================================

def calculate_slice_metrics(slice_df: pd.DataFrame, overall_df: pd.DataFrame) -> dict:
    """
    Calculate metrics for a single slice compared to overall dataset.
    
    Returns:
        dict: Metrics for the slice
    """
    metrics = {
        "count": len(slice_df),
        "proportion": len(slice_df) / len(overall_df),
    }
    
    # Quality score metrics
    if "quality_score" in slice_df.columns:
        metrics["quality_score"] = {
            "mean": float(slice_df["quality_score"].mean()),
            "std": float(slice_df["quality_score"].std()),
            "median": float(slice_df["quality_score"].median()),
        }
    
    # Answer score metrics  
    if "answer_score" in slice_df.columns:
        metrics["answer_score"] = {
            "mean": float(slice_df["answer_score"].mean()),
            "std": float(slice_df["answer_score"].std()),
        }
    
    # Text length metrics
    if "question_body" in slice_df.columns:
        q_lengths = slice_df["question_body"].str.len()
        metrics["question_length"] = {
            "mean": float(q_lengths.mean()),
            "std": float(q_lengths.std()),
        }
    
    if "answer_body" in slice_df.columns:
        a_lengths = slice_df["answer_body"].str.len()
        metrics["answer_length"] = {
            "mean": float(a_lengths.mean()),
            "std": float(a_lengths.std()),
        }
    
    # Complexity distribution
    if "complexity" in slice_df.columns:
        metrics["complexity_distribution"] = slice_df["complexity"].value_counts(normalize=True).to_dict()
    
    return metrics


def detect_bias(slice_metrics: dict, overall_metrics: dict, threshold: float = None) -> dict:
    """
    Detect if a slice shows significant bias compared to overall metrics.
    
    Args:
        slice_metrics: Metrics for the slice
        overall_metrics: Metrics for the overall dataset
        threshold: Difference threshold to flag as biased (default: 0.25 = 25%)
    
    Returns:
        dict: Bias detection results
    """
    threshold = threshold or config.BIAS_THRESHOLD
    bias_results = {
        "is_biased": False,
        "bias_factors": [],
        "severity": "none",
    }
    
    # Check quality score bias - only flag significant differences
    if "quality_score" in slice_metrics and "quality_score" in overall_metrics:
        slice_mean = slice_metrics["quality_score"]["mean"]
        overall_mean = overall_metrics["quality_score"]["mean"]
        
        if overall_mean > 0:
            diff_ratio = abs(slice_mean - overall_mean) / overall_mean
            # Only flag if difference is substantial (> threshold)
            if diff_ratio > threshold:
                bias_results["is_biased"] = True
                direction = "higher" if slice_mean > overall_mean else "lower"
                bias_results["bias_factors"].append({
                    "factor": "quality_score",
                    "slice_value": round(slice_mean, 2),
                    "overall_value": round(overall_mean, 2),
                    "difference_ratio": round(diff_ratio, 3),
                    "direction": direction,
                })
    
    # Check answer length bias - use higher threshold (50% difference)
    if "answer_length" in slice_metrics and "answer_length" in overall_metrics:
        slice_mean = slice_metrics["answer_length"]["mean"]
        overall_mean = overall_metrics["answer_length"]["mean"]
        
        if overall_mean > 0:
            diff_ratio = abs(slice_mean - overall_mean) / overall_mean
            # Higher threshold for length - 50% difference
            if diff_ratio > 0.5:
                bias_results["is_biased"] = True
                direction = "longer" if slice_mean > overall_mean else "shorter"
                bias_results["bias_factors"].append({
                    "factor": "answer_length",
                    "slice_value": round(slice_mean, 2),
                    "overall_value": round(overall_mean, 2),
                    "difference_ratio": round(diff_ratio, 3),
                    "direction": direction,
                })
    
    # Check representation bias - only very severe underrepresentation
    slice_proportion = slice_metrics.get("proportion", 0)
    slice_count = slice_metrics.get("count", 0)
    
    # Flag only if BOTH proportion is tiny AND absolute count is low
    if slice_proportion < 0.005 and slice_count < 20:  # Less than 0.5% AND fewer than 20 samples
        bias_results["bias_factors"].append({
            "factor": "underrepresentation",
            "slice_proportion": round(slice_proportion, 4),
            "slice_count": slice_count,
            "message": "Slice severely underrepresented"
        })
        bias_results["is_biased"] = True
    
    # Determine severity based on number and magnitude of bias factors
    if bias_results["is_biased"]:
        max_diff = max([f.get("difference_ratio", 0) for f in bias_results["bias_factors"]], default=0)
        num_factors = len(bias_results["bias_factors"])
        
        if max_diff > 0.75 or num_factors >= 3:  # 75%+ difference or multiple factors
            bias_results["severity"] = "high"
        elif max_diff > 0.5 or num_factors >= 2:  # 50%+ difference
            bias_results["severity"] = "medium"
        else:
            bias_results["severity"] = "low"
    
    return bias_results


# =============================================================================
# Bias Mitigation Suggestions
# =============================================================================

def suggest_mitigations(bias_report: dict) -> list:
    """
    Suggest mitigation strategies based on detected biases.
    
    Args:
        bias_report: Full bias analysis report
    
    Returns:
        list: Suggested mitigation strategies
    """
    suggestions = []
    
    for slice_name, slice_info in bias_report.get("slice_analysis", {}).items():
        bias_result = slice_info.get("bias_detection", {})
        
        if not bias_result.get("is_biased"):
            continue
        
        for factor in bias_result.get("bias_factors", []):
            factor_type = factor.get("factor")
            
            if factor_type == "underrepresentation":
                suggestions.append({
                    "slice": slice_name,
                    "issue": "Underrepresentation",
                    "strategy": "data_augmentation",
                    "description": f"Consider augmenting data for '{slice_name}' through:"
                                  f" (1) Collecting more examples, "
                                  f"(2) Synthetic data generation, "
                                  f"(3) Oversampling existing examples",
                    "priority": "high" if slice_info["metrics"]["count"] < 50 else "medium"
                })
            
            elif factor_type == "quality_score":
                direction = factor.get("direction")
                if direction == "lower":
                    suggestions.append({
                        "slice": slice_name,
                        "issue": "Lower quality examples",
                        "strategy": "quality_filtering",
                        "description": f"Slice '{slice_name}' has lower quality scores. Consider:"
                                      f" (1) Setting higher quality thresholds for this slice, "
                                      f"(2) Manual review and curation, "
                                      f"(3) Weighted sampling during training",
                        "priority": "medium"
                    })
            
            elif factor_type == "answer_length":
                direction = factor.get("direction")
                suggestions.append({
                    "slice": slice_name,
                    "issue": f"Answers are {direction} than average",
                    "strategy": "length_normalization",
                    "description": f"Consider normalizing answer lengths or using "
                                  f"length-aware training strategies",
                    "priority": "low"
                })
    
    return suggestions


# =============================================================================
# Main Bias Detection Pipeline
# =============================================================================

def run_bias_detection() -> dict:
    """
    Main bias detection function - entry point for Airflow task.
    
    Returns:
        dict: Complete bias analysis report
    """
    logger.info("=" * 60)
    logger.info("STARTING BIAS DETECTION")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Load validated data
        input_path = config.validated_dir / "qa_pairs_validated.json"
        if not input_path.exists():
            input_path = config.processed_dir / "qa_pairs_processed.json"
            logger.warning(f"Validated data not found, using processed data")
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records for bias analysis")
        
        # Calculate overall metrics
        overall_metrics = calculate_slice_metrics(df, df)
        
        # Create slices
        all_slices = {}
        
        # Slice by question type
        all_slices.update(create_slices_by_column(df, "question_type"))
        
        # Slice by complexity
        all_slices.update(create_slices_by_column(df, "complexity"))
        
        # Slice by tags (top tags)
        all_slices.update(create_slices_by_tags(df))
        
        # Slice by error signatures
        all_slices.update(create_slices_by_error_signature(df))
        
        # Slice by infrastructure component
        if "infra_components" in df.columns:
            all_infra = set()
            for comps in df["infra_components"]:
                if isinstance(comps, list):
                    all_infra.update(comps)
            
            for infra in all_infra:
                mask = df["infra_components"].apply(
                    lambda x: infra in x if isinstance(x, list) else False
                )
                slice_df = df[mask]
                if len(slice_df) >= config.SLICE_MIN_SAMPLES:
                    all_slices[f"infra_{infra}"] = slice_df
        
        logger.info(f"Created {len(all_slices)} slices for analysis")
        
        # Analyze each slice
        slice_analysis = {}
        biased_slices = []
        
        for slice_name, slice_df in all_slices.items():
            slice_metrics = calculate_slice_metrics(slice_df, df)
            bias_detection = detect_bias(slice_metrics, overall_metrics)
            
            slice_analysis[slice_name] = {
                "metrics": slice_metrics,
                "bias_detection": bias_detection,
            }
            
            if bias_detection["is_biased"]:
                biased_slices.append(slice_name)
                logger.warning(f"Bias detected in slice '{slice_name}': "
                             f"severity={bias_detection['severity']}")
        
        # Generate report
        report = {
            "analysis_timestamp": start_time.isoformat(),
            "total_records": len(df),
            "total_slices": len(all_slices),
            "biased_slices_count": len(biased_slices),
            "biased_slices": biased_slices,
            "overall_metrics": overall_metrics,
            "slice_analysis": slice_analysis,
        }
        
        # Generate mitigation suggestions
        mitigations = suggest_mitigations(report)
        report["mitigation_suggestions"] = mitigations
        
        # Summary statistics
        report["summary"] = {
            "bias_rate": len(biased_slices) / len(all_slices) if all_slices else 0,
            "high_severity_count": sum(
                1 for s in slice_analysis.values() 
                if s["bias_detection"]["severity"] == "high"
            ),
            "medium_severity_count": sum(
                1 for s in slice_analysis.values() 
                if s["bias_detection"]["severity"] == "medium"
            ),
            "low_severity_count": sum(
                1 for s in slice_analysis.values() 
                if s["bias_detection"]["severity"] == "low"
            ),
        }
        
        # Save report
        report_path = config.REPORTS_DIR / "bias" / "bias_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save metrics file for DVC tracking
        metrics = {
            "total_slices": len(all_slices),
            "biased_slices": len(biased_slices),
            "bias_rate": report["summary"]["bias_rate"],
            "high_severity_count": report["summary"]["high_severity_count"],
            "medium_severity_count": report["summary"]["medium_severity_count"],
            "low_severity_count": report["summary"]["low_severity_count"],
            "mitigation_suggestions_count": len(mitigations),
            "timestamp": datetime.now().isoformat()
        }
        
        metrics_path = config.REPORTS_DIR / "bias" / "bias_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Bias report saved to {report_path}")
        logger.info(f"Found {len(biased_slices)} biased slices out of {len(all_slices)}")
        
        # Log summary
        if mitigations:
            logger.info(f"Generated {len(mitigations)} mitigation suggestions")
            for m in mitigations[:3]:  # Log first 3
                logger.info(f"  - {m['slice']}: {m['issue']} ({m['priority']} priority)")
        
        report["status"] = "success"
        report["end_time"] = datetime.now().isoformat()
        report["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        
        return report
        
    except Exception as e:
        logger.error(f"Bias detection failed: {e}")
        raise


if __name__ == "__main__":
    run_bias_detection()