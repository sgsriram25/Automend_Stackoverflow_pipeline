"""
AutoMend Data Pipeline DAG
===========================
Airflow DAG for orchestrating the complete data pipeline from
acquisition through preprocessing, validation, and bias detection.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# Add scripts directory to path - handle both local and Docker paths
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if not SCRIPTS_DIR.exists():
    SCRIPTS_DIR = Path("/opt/airflow/scripts")
sys.path.insert(0, str(SCRIPTS_DIR))

# Set environment variable for config to find data
os.environ.setdefault("AIRFLOW_HOME", str(Path(__file__).parent.parent))

# Import pipeline modules - with error handling for Airflow
try:
    from data_acquisition import run_acquisition
    from data_preprocessing import run_preprocessing
    from data_validation import run_validation
    from bias_detection import run_bias_detection
    from alerts import (
        alert_pipeline_start, alert_pipeline_success, 
        alert_pipeline_failure, alert_validation_failure,
        alert_anomalies_detected, alert_bias_detected
    )
    from config import config
    MODULES_LOADED = True
except ImportError as e:
    # Log import error but don't fail DAG parsing
    import logging
    logging.error(f"Failed to import pipeline modules: {e}")
    MODULES_LOADED = False
    
    # Create dummy config for DAG parsing
    class DummyConfig:
        ALERT_EMAIL = None
        validated_dir = Path("/opt/airflow/data/validated")
        processed_dir = Path("/opt/airflow/data/processed")
        training_dir = Path("/opt/airflow/data/training")
    config = DummyConfig()
    
    # Create dummy functions that raise clear errors
    def run_acquisition(**kwargs): 
        raise ImportError(f"Module not loaded. Check PYTHONPATH includes scripts directory.")
    def run_preprocessing(**kwargs): 
        raise ImportError("Module not loaded")
    def run_validation(**kwargs): 
        raise ImportError("Module not loaded")
    def run_bias_detection(**kwargs): 
        raise ImportError("Module not loaded")
    def alert_pipeline_start(*args): pass
    def alert_pipeline_success(*args): pass
    def alert_pipeline_failure(*args): pass
    def alert_validation_failure(*args): pass
    def alert_anomalies_detected(*args): pass
    def alert_bias_detected(*args): pass


# =============================================================================
# DAG Configuration
# =============================================================================

# Get email safely
alert_email = getattr(config, 'ALERT_EMAIL', None)

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': [alert_email] if alert_email else [],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

dag = DAG(
    'automend_data_pipeline',
    default_args=default_args,
    description='AutoMend StackOverflow Data Pipeline for MLOps Training',
    schedule_interval='@daily',  # Run daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'data-pipeline', 'stackoverflow'],
    doc_md="""
    # AutoMend Data Pipeline
    
    This DAG orchestrates the complete data pipeline:
    1. **Data Acquisition**: Fetch Q&A data from StackOverflow API or CSV
    2. **Preprocessing**: Clean and transform data, extract features
    3. **Validation**: Validate schema, quality, detect anomalies
    4. **Bias Detection**: Analyze data slices for bias
    5. **Training Data**: Generate final training dataset
    
    ## Monitoring
    - Check Gantt chart for bottlenecks
    - Alerts sent to Slack on failures
    - All logs stored in /logs directory
    """,
)


# =============================================================================
# Task Functions
# =============================================================================

def task_start_pipeline(**context):
    """Initialize pipeline and send start notification."""
    dag_id = context['dag'].dag_id
    run_id = context['run_id']
    
    alert_pipeline_start(dag_id, run_id)
    
    return {
        "start_time": datetime.now().isoformat(),
        "dag_id": dag_id,
        "run_id": run_id
    }


def task_acquire_data(use_csv: bool = True, **context):
    """
    Acquire data from StackOverflow API or local CSV.
    
    Args:
        use_csv: If True, use local CSV files instead of API
    """
    try:
        stats = run_acquisition(use_csv=use_csv)
        
        # Push stats to XCom for downstream tasks
        context['ti'].xcom_push(key='acquisition_stats', value=stats)
        
        if stats.get("status") != "success":
            raise Exception(f"Acquisition failed: {stats.get('error')}")
        
        return stats
        
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id,
            context['run_id'],
            'acquire_data',
            str(e)
        )
        raise


def task_preprocess_data(**context):
    """Clean and transform raw data."""
    try:
        stats = run_preprocessing()
        
        context['ti'].xcom_push(key='preprocessing_stats', value=stats)
        
        if stats.get("status") != "success":
            raise Exception(f"Preprocessing failed: {stats.get('error')}")
        
        return stats
        
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id,
            context['run_id'],
            'preprocess_data',
            str(e)
        )
        raise


def task_validate_data(**context):
    """Validate data quality and schema."""
    try:
        results = run_validation()
        
        context['ti'].xcom_push(key='validation_results', value=results)
        
        # Check for anomalies
        anomaly_count = results.get("statistics", {}).get("total_anomalies", 0)
        if anomaly_count > 0:
            alert_anomalies_detected(
                anomaly_count,
                ["outlier"]  # Simplified for now
            )
        
        return results
        
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id,
            context['run_id'],
            'validate_data',
            str(e)
        )
        raise


def task_check_validation(**context):
    """
    Branch based on validation results.
    Returns task_id to execute next.
    """
    ti = context['ti']
    validation_results = ti.xcom_pull(key='validation_results', task_ids='validate_data')
    
    if validation_results and validation_results.get("is_valid"):
        return 'detect_bias'
    else:
        # Send alert and continue to failure handling
        alert_validation_failure(validation_results or {})
        return 'handle_validation_failure'


def task_handle_validation_failure(**context):
    """Handle validation failure - log and potentially halt."""
    ti = context['ti']
    validation_results = ti.xcom_pull(key='validation_results', task_ids='validate_data')
    
    # Log detailed failure info
    failed_checks = validation_results.get("failed", []) if validation_results else []
    
    error_msg = f"Validation failed with {len(failed_checks)} issues"
    
    # You could choose to raise an exception here to fail the DAG
    # or continue with reduced data
    # For now, we'll log and continue
    print(f"WARNING: {error_msg}")
    print(f"Failed checks: {json.dumps(failed_checks, indent=2)}")
    
    return {"status": "validation_failed", "issues": len(failed_checks)}


def task_detect_bias(**context):
    """Run bias detection on validated data."""
    try:
        report = run_bias_detection()
        
        context['ti'].xcom_push(key='bias_report', value=report)
        
        # Alert if significant bias detected
        biased_slices = report.get("biased_slices", [])
        if biased_slices:
            summary = report.get("summary", {})
            alert_bias_detected(biased_slices, {
                "high": summary.get("high_severity_count", 0),
                "medium": summary.get("medium_severity_count", 0),
                "low": summary.get("low_severity_count", 0),
            })
        
        return report
        
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id,
            context['run_id'],
            'detect_bias',
            str(e)
        )
        raise


def task_generate_training_data(**context):
    """Generate final training dataset."""
    try:
        # Load validated data
        input_path = config.validated_dir / "qa_pairs_validated.json"
        if not input_path.exists():
            input_path = config.processed_dir / "qa_pairs_processed.json"
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert to training format
        training_data = []
        for record in data:
            training_data.append({
                "messages": [
                    {"role": "user", "content": record.get("question_body", "")},
                    {"role": "assistant", "content": record.get("answer_body", "")}
                ],
                "metadata": {
                    "question_id": record.get("question_id"),
                    "tags": record.get("tags", []),
                    "error_signatures": record.get("error_signatures", []),
                    "infra_components": record.get("infra_components", []),
                    "quality_score": record.get("quality_score", 0),
                }
            })
        
        # Save training data
        output_path = config.training_dir / "training_data.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2)
        
        # Also save as JSONL
        jsonl_path = config.training_dir / "training_data.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for example in training_data:
                f.write(json.dumps(example) + "\n")
        
        stats = {
            "total_examples": len(training_data),
            "output_path": str(output_path),
            "jsonl_path": str(jsonl_path),
        }
        
        context['ti'].xcom_push(key='training_stats', value=stats)
        
        return stats
        
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id,
            context['run_id'],
            'generate_training_data',
            str(e)
        )
        raise


def task_end_pipeline(**context):
    """Finalize pipeline and send completion notification."""
    ti = context['ti']
    
    # Gather all stats
    acquisition_stats = ti.xcom_pull(key='acquisition_stats', task_ids='acquire_data') or {}
    preprocessing_stats = ti.xcom_pull(key='preprocessing_stats', task_ids='preprocess_data') or {}
    training_stats = ti.xcom_pull(key='training_stats', task_ids='generate_training_data') or {}
    
    # Calculate duration
    start_info = ti.xcom_pull(task_ids='start_pipeline') or {}
    start_time = datetime.fromisoformat(start_info.get("start_time", datetime.now().isoformat()))
    duration = (datetime.now() - start_time).total_seconds()
    
    # Send success notification
    alert_pipeline_success(
        context['dag'].dag_id,
        context['run_id'],
        duration,
        {
            "records_acquired": acquisition_stats.get("total_pairs", "N/A"),
            "records_processed": preprocessing_stats.get("output_records", "N/A"),
            "training_examples": training_stats.get("total_examples", "N/A"),
        }
    )
    
    return {
        "status": "success",
        "duration_seconds": duration,
        "training_examples": training_stats.get("total_examples", 0)
    }


# =============================================================================
# Define Tasks
# =============================================================================

with dag:
    
    # Start task
    start = PythonOperator(
        task_id='start_pipeline',
        python_callable=task_start_pipeline,
        doc_md="Initialize pipeline and send start notification"
    )
    
    # Data acquisition
    acquire = PythonOperator(
        task_id='acquire_data',
        python_callable=task_acquire_data,
        op_kwargs={'use_csv': True},  # Set to False to use API
        doc_md="Fetch data from StackOverflow API or CSV files"
    )
    
    # Preprocessing
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=task_preprocess_data,
        doc_md="Clean, transform, and extract features from raw data"
    )
    
    # Validation
    validate = PythonOperator(
        task_id='validate_data',
        python_callable=task_validate_data,
        doc_md="Validate data schema, quality, and detect anomalies"
    )
    
    # Validation check (branching)
    check_validation = BranchPythonOperator(
        task_id='check_validation',
        python_callable=task_check_validation,
        doc_md="Branch based on validation results"
    )
    
    # Handle validation failure
    handle_failure = PythonOperator(
        task_id='handle_validation_failure',
        python_callable=task_handle_validation_failure,
        doc_md="Handle validation failures"
    )
    
    # Bias detection
    bias_detection = PythonOperator(
        task_id='detect_bias',
        python_callable=task_detect_bias,
        doc_md="Analyze data slices for potential bias"
    )
    
    # Generate training data
    generate_training = PythonOperator(
        task_id='generate_training_data',
        python_callable=task_generate_training_data,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        doc_md="Generate final training dataset"
    )
    
    # End task
    end = PythonOperator(
        task_id='end_pipeline',
        python_callable=task_end_pipeline,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        doc_md="Finalize pipeline and send completion notification"
    )
    
    # Join point for branches
    join = EmptyOperator(
        task_id='join',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )
    
    # =============================================================================
    # Define Dependencies (DAG Structure)
    # =============================================================================
    
    # Main flow
    start >> acquire >> preprocess >> validate >> check_validation
    
    # Branching after validation check
    check_validation >> [bias_detection, handle_failure]
    
    # Both branches lead to join
    bias_detection >> join
    handle_failure >> join
    
    # Continue to training data generation
    join >> generate_training >> end