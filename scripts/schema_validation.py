"""
Schema Validation Module (Great Expectations)
===============================================
Automated data schema and statistics generation using Great Expectations.
Validates data quality over time and generates data documentation.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

# Try to import Great Expectations (optional dependency)
try:
    import great_expectations as gx
    from great_expectations.core.expectation_configuration import ExpectationConfiguration
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False

from config import config

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'schema_validation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Schema Definition
# =============================================================================

EXPECTED_SCHEMA = {
    "columns": {
        "question_id": {
            "type": "int64",
            "nullable": False,
            "unique": True,
            "min_value": 1
        },
        "title": {
            "type": "object",
            "nullable": False,
            "min_length": 5,
            "max_length": 500
        },
        "question_body": {
            "type": "object",
            "nullable": False,
            "min_length": 20
        },
        "answer_body": {
            "type": "object",
            "nullable": False,
            "min_length": 50
        },
        "tags": {
            "type": "object",  # list stored as object
            "nullable": False
        },
        "score": {
            "type": "int64",
            "nullable": False,
            "min_value": 0
        },
        "answer_score": {
            "type": "int64",
            "nullable": True,
            "min_value": 0
        },
        "quality_score": {
            "type": "float64",
            "nullable": False,
            "min_value": 0
        }
    },
    "required_columns": [
        "question_id", "title", "question_body", "answer_body",
        "tags", "score", "quality_score"
    ],
    "row_count": {
        "min": 100,
        "max": 1000000
    }
}


# =============================================================================
# Statistics Generation
# =============================================================================

def generate_column_statistics(df: pd.DataFrame, column: str) -> dict:
    """Generate detailed statistics for a single column."""
    stats = {
        "name": column,
        "dtype": str(df[column].dtype),
        "count": int(df[column].count()),
        "null_count": int(df[column].isna().sum()),
        "null_percentage": float(df[column].isna().mean() * 100),
    }
    
    if pd.api.types.is_numeric_dtype(df[column]):
        stats.update({
            "mean": float(df[column].mean()) if not df[column].isna().all() else None,
            "std": float(df[column].std()) if not df[column].isna().all() else None,
            "min": float(df[column].min()) if not df[column].isna().all() else None,
            "max": float(df[column].max()) if not df[column].isna().all() else None,
            "median": float(df[column].median()) if not df[column].isna().all() else None,
            "q25": float(df[column].quantile(0.25)) if not df[column].isna().all() else None,
            "q75": float(df[column].quantile(0.75)) if not df[column].isna().all() else None,
        })
    elif pd.api.types.is_string_dtype(df[column]) or df[column].dtype == 'object':
        # For string columns
        non_null = df[column].dropna()
        if len(non_null) > 0:
            try:
                lengths = non_null.astype(str).str.len()
                stats.update({
                    "unique_count": int(non_null.nunique()),
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "mean_length": float(lengths.mean()),
                    "most_common": non_null.value_counts().head(5).to_dict() if non_null.nunique() < 100 else None
                })
            except Exception:
                pass
    
    return stats


def generate_data_statistics(df: pd.DataFrame) -> dict:
    """Generate comprehensive statistics for the entire dataset."""
    logger.info("Generating data statistics...")
    
    stats = {
        "generated_at": datetime.now().isoformat(),
        "dataset_info": {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "column_statistics": {}
    }
    
    for col in df.columns:
        try:
            stats["column_statistics"][col] = generate_column_statistics(df, col)
        except Exception as e:
            logger.warning(f"Failed to generate stats for column {col}: {e}")
            stats["column_statistics"][col] = {"error": str(e)}
    
    return stats


# =============================================================================
# Schema Validation (without Great Expectations)
# =============================================================================

def validate_schema_simple(df: pd.DataFrame) -> dict:
    """Validate DataFrame against expected schema (no GX dependency)."""
    logger.info("Validating schema (simple mode)...")
    
    results = {
        "valid": True,
        "timestamp": datetime.now().isoformat(),
        "checks": [],
        "summary": {"passed": 0, "failed": 0, "warnings": 0}
    }
    
    # Check required columns
    missing_cols = set(EXPECTED_SCHEMA["required_columns"]) - set(df.columns)
    if missing_cols:
        results["checks"].append({
            "check": "required_columns",
            "status": "FAILED",
            "message": f"Missing columns: {missing_cols}"
        })
        results["valid"] = False
        results["summary"]["failed"] += 1
    else:
        results["checks"].append({
            "check": "required_columns",
            "status": "PASSED",
            "message": "All required columns present"
        })
        results["summary"]["passed"] += 1
    
    # Check row count
    row_count = len(df)
    if row_count < EXPECTED_SCHEMA["row_count"]["min"]:
        results["checks"].append({
            "check": "min_row_count",
            "status": "FAILED",
            "message": f"Row count {row_count} below minimum {EXPECTED_SCHEMA['row_count']['min']}"
        })
        results["valid"] = False
        results["summary"]["failed"] += 1
    else:
        results["checks"].append({
            "check": "min_row_count",
            "status": "PASSED",
            "message": f"Row count {row_count} meets minimum"
        })
        results["summary"]["passed"] += 1
    
    # Check each column
    for col_name, col_schema in EXPECTED_SCHEMA["columns"].items():
        if col_name not in df.columns:
            continue
        
        col = df[col_name]
        
        # Check nullability
        if not col_schema.get("nullable", True) and col.isna().any():
            null_count = col.isna().sum()
            results["checks"].append({
                "check": f"{col_name}_not_null",
                "status": "FAILED",
                "message": f"Column '{col_name}' has {null_count} null values but should not be nullable"
            })
            results["valid"] = False
            results["summary"]["failed"] += 1
        else:
            results["checks"].append({
                "check": f"{col_name}_not_null",
                "status": "PASSED",
                "message": f"Column '{col_name}' null check passed"
            })
            results["summary"]["passed"] += 1
        
        # Check uniqueness
        if col_schema.get("unique", False):
            duplicates = col.duplicated().sum()
            if duplicates > 0:
                results["checks"].append({
                    "check": f"{col_name}_unique",
                    "status": "FAILED",
                    "message": f"Column '{col_name}' has {duplicates} duplicate values"
                })
                results["valid"] = False
                results["summary"]["failed"] += 1
            else:
                results["checks"].append({
                    "check": f"{col_name}_unique",
                    "status": "PASSED",
                    "message": f"Column '{col_name}' values are unique"
                })
                results["summary"]["passed"] += 1
        
        # Check min value for numeric columns
        if "min_value" in col_schema and pd.api.types.is_numeric_dtype(col):
            min_val = col.min()
            if min_val < col_schema["min_value"]:
                results["checks"].append({
                    "check": f"{col_name}_min_value",
                    "status": "FAILED",
                    "message": f"Column '{col_name}' min value {min_val} below {col_schema['min_value']}"
                })
                results["summary"]["failed"] += 1
            else:
                results["checks"].append({
                    "check": f"{col_name}_min_value",
                    "status": "PASSED",
                    "message": f"Column '{col_name}' min value check passed"
                })
                results["summary"]["passed"] += 1
    
    return results


# =============================================================================
# Great Expectations Integration (if available)
# =============================================================================

def validate_with_great_expectations(df: pd.DataFrame) -> dict:
    """Validate DataFrame using Great Expectations."""
    if not GX_AVAILABLE:
        logger.warning("Great Expectations not installed, using simple validation")
        return validate_schema_simple(df)
    
    logger.info("Validating with Great Expectations...")
    
    try:
        # Create a GX context
        context = gx.get_context()
        
        # Create a data source from the DataFrame
        data_source = context.sources.add_pandas("pandas_source")
        data_asset = data_source.add_dataframe_asset(name="qa_data")
        
        # Build batch request
        batch_request = data_asset.build_batch_request(dataframe=df)
        
        # Create expectation suite
        suite_name = "automend_qa_validation"
        
        # Define expectations
        expectations = [
            gx.expectations.ExpectTableRowCountToBeBetween(min_value=100, max_value=1000000),
            gx.expectations.ExpectColumnToExist(column="question_id"),
            gx.expectations.ExpectColumnToExist(column="question_body"),
            gx.expectations.ExpectColumnToExist(column="answer_body"),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="question_id"),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="question_body"),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="answer_body"),
            gx.expectations.ExpectColumnValuesToBeUnique(column="question_id"),
            gx.expectations.ExpectColumnValuesToBeBetween(column="score", min_value=0),
            gx.expectations.ExpectColumnValuesToBeBetween(column="quality_score", min_value=0),
        ]
        
        # Run validation
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
        
        for expectation in expectations:
            validator.expect(expectation)
        
        results = validator.validate()
        
        return {
            "valid": results.success,
            "timestamp": datetime.now().isoformat(),
            "statistics": results.statistics,
            "results": [
                {
                    "expectation": str(r.expectation_config.expectation_type),
                    "success": r.success,
                    "result": r.result
                }
                for r in results.results
            ]
        }
        
    except Exception as e:
        logger.error(f"Great Expectations validation failed: {e}")
        logger.info("Falling back to simple validation")
        return validate_schema_simple(df)


# =============================================================================
# Main Function
# =============================================================================

def run_schema_validation() -> dict:
    """Main schema validation function - entry point for Airflow/DVC."""
    logger.info("=" * 60)
    logger.info("STARTING SCHEMA VALIDATION")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Load data
        input_path = config.processed_dir / "qa_pairs_processed.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records")
        
        # Generate statistics
        statistics = generate_data_statistics(df)
        
        # Save statistics
        stats_path = config.REPORTS_DIR / "statistics" / "data_statistics.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, default=str)
        logger.info(f"Statistics saved to {stats_path}")
        
        # Validate schema
        if GX_AVAILABLE:
            validation_results = validate_with_great_expectations(df)
        else:
            validation_results = validate_schema_simple(df)
        
        # Save validation results
        validation_path = config.REPORTS_DIR / "validation" / "schema_validation.json"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(validation_results, f, indent=2, default=str)
        logger.info(f"Validation results saved to {validation_path}")
        
        # Create metrics file for DVC
        metrics = {
            "schema_valid": validation_results["valid"],
            "checks_passed": validation_results.get("summary", {}).get("passed", 0),
            "checks_failed": validation_results.get("summary", {}).get("failed", 0),
            "row_count": len(df),
            "column_count": len(df.columns),
            "timestamp": datetime.now().isoformat()
        }
        
        metrics_path = config.REPORTS_DIR / "validation" / "validation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Schema validation: {'PASSED' if validation_results['valid'] else 'FAILED'}")
        
        return {
            "status": "success",
            "valid": validation_results["valid"],
            "statistics": statistics,
            "validation": validation_results,
            "duration_seconds": (datetime.now() - start_time).total_seconds()
        }
        
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise


if __name__ == "__main__":
    run_schema_validation()