"""
Data Preprocessing Module
==========================
Cleans, transforms, and engineers features from raw Q&A data.
Modular design for easy testing and updates.
"""

import json
import re
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np

from config import config, ERROR_PATTERNS, INFRA_PATTERNS

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
        logging.FileHandler(config.LOGS_DIR / 'preprocessing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Text Cleaning Functions
# =============================================================================

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_code_blocks(text: str) -> tuple:
    """
    Extract code blocks from text and return (text_without_code, code_blocks).
    Preserves structure for analysis.
    """
    code_pattern = r'```[\s\S]*?```|`[^`]+`'
    code_blocks = re.findall(code_pattern, text)
    text_without_code = re.sub(code_pattern, ' [CODE] ', text)
    return normalize_whitespace(text_without_code), code_blocks


def clean_text(text: str, remove_code: bool = False) -> str:
    """
    Clean text with configurable options.
    
    Args:
        text: Input text to clean
        remove_code: If True, removes code blocks entirely
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    
    # Handle code blocks
    if remove_code:
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    return text


def calculate_text_metrics(text: str) -> dict:
    """Calculate various text metrics."""
    if not text:
        return {"char_count": 0, "word_count": 0, "sentence_count": 0, "code_block_count": 0}
    
    _, code_blocks = extract_code_blocks(text)
    
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "sentence_count": len(re.findall(r'[.!?]+', text)),
        "code_block_count": len(code_blocks),
    }


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_error_signatures(text: str) -> list:
    """Extract error pattern signatures from text."""
    found = []
    for name, pattern in ERROR_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(name)
    return found


def extract_infra_components(text: str) -> list:
    """Identify infrastructure components mentioned in text."""
    found = []
    for name, pattern in INFRA_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(name)
    return found


def calculate_quality_score(row: dict) -> float:
    """
    Calculate a quality score for ranking examples.
    Higher score = better quality training example.
    """
    score = row.get("score", 0)
    answer_score = row.get("answer_score", 0)
    view_count = row.get("view_count", 0)
    
    # Text quality indicators
    q_metrics = calculate_text_metrics(row.get("question_body", ""))
    a_metrics = calculate_text_metrics(row.get("answer_body", ""))
    
    # Weighted formula
    quality = (
        score * 2.0 +                              # Question upvotes
        answer_score * 3.0 +                       # Answer upvotes (more important)
        view_count * 0.001 +                       # Popularity
        min(q_metrics["word_count"] / 50, 2) +     # Question length (cap at 2)
        min(a_metrics["word_count"] / 100, 3) +    # Answer length (cap at 3)
        a_metrics["code_block_count"] * 0.5        # Code examples
    )
    
    return round(quality, 2)


def classify_complexity(text: str) -> str:
    """Classify the complexity of the question/answer."""
    metrics = calculate_text_metrics(text)
    word_count = metrics["word_count"]
    code_count = metrics["code_block_count"]
    
    if word_count < 50 and code_count == 0:
        return "simple"
    elif word_count < 200 or code_count <= 1:
        return "moderate"
    else:
        return "complex"


def detect_question_type(text: str) -> str:
    """Detect the type of question being asked."""
    text_lower = text.lower()
    
    if re.search(r'how (do|can|to|should)', text_lower):
        return "how_to"
    elif re.search(r'why (is|does|did|am|are)', text_lower):
        return "explanation"
    elif re.search(r'what (is|are|does)', text_lower):
        return "definition"
    elif re.search(r'(fix|solve|resolve|debug)', text_lower):
        return "troubleshooting"
    elif re.search(r'(error|exception|fail|crash)', text_lower):
        return "error_resolution"
    elif re.search(r'(best practice|recommend|should i)', text_lower):
        return "best_practice"
    else:
        return "general"


# =============================================================================
# Main Preprocessing Pipeline
# =============================================================================

def preprocess_single_record(record: dict) -> Optional[dict]:
    """
    Preprocess a single Q&A record.
    Returns None if record should be filtered out.
    """
    # Validate required fields
    required_fields = ["question_id", "question_body", "answer_body"]
    if not all(record.get(f) for f in required_fields):
        return None
    
    question_body = clean_text(record.get("question_body", ""))
    answer_body = clean_text(record.get("answer_body", ""))
    combined_text = f"{record.get('title', '')} {question_body} {answer_body}"
    
    # Filter out very short content
    if len(question_body) < 20 or len(answer_body) < 50:
        return None
    
    # Build processed record
    processed = {
        # Identifiers
        "question_id": record["question_id"],
        
        # Cleaned text
        "title": clean_text(record.get("title", "")),
        "question_body": question_body,
        "answer_body": answer_body,
        
        # Original metadata
        "tags": record.get("tags", []),
        "score": record.get("score", 0),
        "answer_score": record.get("answer_score", 0),
        "view_count": record.get("view_count", 0),
        
        # Extracted features
        "error_signatures": extract_error_signatures(combined_text),
        "infra_components": extract_infra_components(combined_text),
        "question_type": detect_question_type(question_body),
        "complexity": classify_complexity(combined_text),
        
        # Metrics
        "question_metrics": calculate_text_metrics(question_body),
        "answer_metrics": calculate_text_metrics(answer_body),
        "quality_score": calculate_quality_score(record),
        
        # Processing metadata
        "processed_at": datetime.now().isoformat(),
    }
    
    return processed


def run_preprocessing() -> dict:
    """
    Main preprocessing function - entry point for Airflow task.
    
    Returns:
        dict: Statistics about the preprocessing
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    stats = {"start_time": start_time.isoformat()}
    
    try:
        # Load raw data
        input_path = config.raw_dir / "qa_pairs_raw.json"
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        logger.info(f"Loaded {len(raw_data)} raw records")
        stats["input_records"] = len(raw_data)
        
        # Process each record
        processed_data = []
        filtered_reasons = Counter()
        
        for record in raw_data:
            result = preprocess_single_record(record)
            if result:
                processed_data.append(result)
            else:
                filtered_reasons["invalid_or_short"] += 1
        
        logger.info(f"Processed {len(processed_data)} records ({len(raw_data) - len(processed_data)} filtered)")
        
        # Sort by quality score
        processed_data.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Save processed data
        output_path = config.processed_dir / "qa_pairs_processed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2)
        
        # Also save as DataFrame for analysis
        df = pd.DataFrame(processed_data)
        df.to_csv(config.processed_dir / "qa_pairs_processed.csv", index=False)
        
        # Calculate statistics
        stats.update({
            "output_records": len(processed_data),
            "filtered_count": len(raw_data) - len(processed_data),
            "filter_reasons": dict(filtered_reasons),
            "output_file": str(output_path),
            "status": "success"
        })
        
        # Feature distribution stats
        if processed_data:
            stats["feature_stats"] = {
                "avg_quality_score": np.mean([r["quality_score"] for r in processed_data]),
                "error_signature_dist": Counter(
                    sig for r in processed_data for sig in r["error_signatures"]
                ),
                "infra_component_dist": Counter(
                    comp for r in processed_data for comp in r["infra_components"]
                ),
                "question_type_dist": Counter(r["question_type"] for r in processed_data),
                "complexity_dist": Counter(r["complexity"] for r in processed_data),
                "tag_dist": Counter(tag for r in processed_data for tag in r["tags"]),
            }
            
            # Convert Counters to dicts for JSON serialization
            for key in ["error_signature_dist", "infra_component_dist", 
                       "question_type_dist", "complexity_dist", "tag_dist"]:
                stats["feature_stats"][key] = dict(stats["feature_stats"][key])
        
        logger.info(f"Saved processed data to {output_path}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        stats["status"] = "failed"
        stats["error"] = str(e)
        raise
    
    finally:
        stats["end_time"] = datetime.now().isoformat()
        stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        
        # Save stats
        stats_path = config.LOGS_DIR / "preprocessing_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Preprocessing completed in {stats['duration_seconds']:.1f}s")
    
    return stats


if __name__ == "__main__":
    run_preprocessing()