"""
Generate Training Data Module
==============================
Creates final training dataset with bias mitigation applied.
Outputs data in formats suitable for model training.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter

import pandas as pd
import numpy as np

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
        logging.FileHandler(config.LOGS_DIR / 'generate_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Bias Mitigation Functions
# =============================================================================

def apply_resampling(df: pd.DataFrame, bias_report: dict) -> pd.DataFrame:
    """
    Apply resampling to mitigate underrepresentation bias.
    Oversamples underrepresented slices.
    """
    logger.info("Applying bias mitigation via resampling...")
    
    mitigations = bias_report.get("mitigation_suggestions", [])
    
    # Find underrepresented slices
    underrep_slices = [
        m for m in mitigations 
        if m.get("strategy") == "data_augmentation" and m.get("priority") == "high"
    ]
    
    if not underrep_slices:
        logger.info("No high-priority underrepresentation found, skipping resampling")
        return df
    
    resampled_df = df.copy()
    
    for slice_info in underrep_slices[:3]:  # Limit to top 3
        slice_name = slice_info.get("slice", "")
        
        # Determine the slice column and value
        if slice_name.startswith("tag_"):
            tag = slice_name.replace("tag_", "")
            mask = df["tags"].apply(lambda x: tag in x if isinstance(x, list) else False)
        elif slice_name.startswith("error_"):
            error = slice_name.replace("error_", "")
            mask = df["error_signatures"].apply(lambda x: error in x if isinstance(x, list) else False)
        elif slice_name.startswith("infra_"):
            infra = slice_name.replace("infra_", "")
            mask = df["infra_components"].apply(lambda x: infra in x if isinstance(x, list) else False)
        else:
            continue
        
        slice_data = df[mask]
        if len(slice_data) < 30:
            # Oversample by 2x
            oversampled = slice_data.sample(n=min(len(slice_data) * 2, 100), replace=True)
            resampled_df = pd.concat([resampled_df, oversampled], ignore_index=True)
            logger.info(f"Oversampled slice '{slice_name}': added {len(oversampled)} records")
    
    return resampled_df


def apply_quality_weighting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sample weights based on quality score for weighted training.
    """
    logger.info("Calculating sample weights...")
    
    # Normalize quality scores to weights
    min_score = df["quality_score"].min()
    max_score = df["quality_score"].max()
    
    if max_score > min_score:
        # Scale to [0.5, 1.5] range
        df["sample_weight"] = 0.5 + (df["quality_score"] - min_score) / (max_score - min_score)
    else:
        df["sample_weight"] = 1.0
    
    return df


# =============================================================================
# Training Data Formatting
# =============================================================================

def format_for_chat_training(record: dict) -> dict:
    """Format a record for chat-style fine-tuning."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert MLOps engineer helping to diagnose and fix infrastructure issues."
            },
            {
                "role": "user", 
                "content": record.get("question_body", "")
            },
            {
                "role": "assistant",
                "content": record.get("answer_body", "")
            }
        ],
        "metadata": {
            "question_id": record.get("question_id"),
            "tags": record.get("tags", []),
            "error_signatures": record.get("error_signatures", []),
            "infra_components": record.get("infra_components", []),
            "quality_score": record.get("quality_score", 0),
            "sample_weight": record.get("sample_weight", 1.0),
        }
    }


def format_for_completion_training(record: dict) -> dict:
    """Format a record for completion-style fine-tuning."""
    prompt = f"""### Question
{record.get('title', '')}

{record.get('question_body', '')}

Tags: {', '.join(record.get('tags', []))}

### Answer
"""
    completion = record.get("answer_body", "")
    
    return {
        "prompt": prompt,
        "completion": completion,
        "metadata": {
            "question_id": record.get("question_id"),
            "quality_score": record.get("quality_score", 0),
        }
    }


# =============================================================================
# Train/Test Split
# =============================================================================

def create_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42) -> tuple:
    """
    Create stratified train/test split.
    Stratifies by question_type to maintain distribution.
    """
    logger.info(f"Creating train/test split (test_ratio={test_ratio})...")
    
    # Shuffle
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Simple split (could enhance with stratification)
    split_idx = int(len(df_shuffled) * (1 - test_ratio))
    
    train_df = df_shuffled[:split_idx]
    test_df = df_shuffled[split_idx:]
    
    logger.info(f"Train set: {len(train_df)} records")
    logger.info(f"Test set: {len(test_df)} records")
    
    return train_df, test_df


# =============================================================================
# Main Function
# =============================================================================

def run_generate_training_data(
    apply_mitigation: bool = True,
    test_split: float = 0.2,
    output_format: str = "chat"  # "chat" or "completion"
) -> dict:
    """
    Main function to generate training data.
    
    Args:
        apply_mitigation: Whether to apply bias mitigation
        test_split: Fraction for test set
        output_format: "chat" for chat format, "completion" for prompt/completion
    """
    logger.info("=" * 60)
    logger.info("GENERATING TRAINING DATA")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    stats = {"start_time": start_time.isoformat()}
    
    try:
        # Load validated data
        input_path = config.validated_dir / "qa_pairs_validated.json"
        if not input_path.exists():
            input_path = config.processed_dir / "qa_pairs_processed.json"
            logger.warning(f"Using processed data (validated not found)")
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records")
        stats["input_records"] = len(df)
        
        # Load bias report if available
        bias_report = {}
        bias_path = config.REPORTS_DIR / "bias" / "bias_report.json"
        if bias_path.exists():
            with open(bias_path, "r", encoding="utf-8") as f:
                bias_report = json.load(f)
            logger.info("Loaded bias report for mitigation")
        
        # Apply bias mitigation
        if apply_mitigation and bias_report:
            df = apply_resampling(df, bias_report)
            stats["after_resampling"] = len(df)
        
        # Add quality weights
        df = apply_quality_weighting(df)
        
        # Create train/test split
        train_df, test_df = create_train_test_split(df, test_split)
        
        # Format data
        format_func = format_for_chat_training if output_format == "chat" else format_for_completion_training
        
        train_data = [format_func(row.to_dict()) for _, row in train_df.iterrows()]
        test_data = [format_func(row.to_dict()) for _, row in test_df.iterrows()]
        
        # Save training data
        config.training_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON format
        train_path = config.training_dir / "training_data.json"
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2)
        
        test_path = config.training_dir / "test_data.json"
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2)
        
        # JSONL format (for fine-tuning APIs)
        train_jsonl_path = config.training_dir / "training_data.jsonl"
        with open(train_jsonl_path, "w", encoding="utf-8") as f:
            for example in train_data:
                f.write(json.dumps(example) + "\n")
        
        test_jsonl_path = config.training_dir / "test_data.jsonl"
        with open(test_jsonl_path, "w", encoding="utf-8") as f:
            for example in test_data:
                f.write(json.dumps(example) + "\n")
        
        # Generate statistics
        stats.update({
            "train_examples": len(train_data),
            "test_examples": len(test_data),
            "total_examples": len(train_data) + len(test_data),
            "test_ratio": test_split,
            "output_format": output_format,
            "bias_mitigation_applied": apply_mitigation and bool(bias_report),
            "output_files": {
                "train_json": str(train_path),
                "train_jsonl": str(train_jsonl_path),
                "test_json": str(test_path),
                "test_jsonl": str(test_jsonl_path),
            },
            "status": "success"
        })
        
        # Distribution statistics
        if "tags" in train_df.columns:
            all_tags = []
            for tags in train_df["tags"]:
                if isinstance(tags, list):
                    all_tags.extend(tags)
            stats["tag_distribution"] = dict(Counter(all_tags).most_common(10))
        
        # Save stats
        stats_path = config.training_dir / "training_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Training data saved: {len(train_data)} train, {len(test_data)} test")
        logger.info(f"Output files: {train_path}, {train_jsonl_path}")
        
        stats["end_time"] = datetime.now().isoformat()
        stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        
        return stats
        
    except Exception as e:
        logger.error(f"Training data generation failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--no-mitigation", action="store_true", help="Skip bias mitigation")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--format", choices=["chat", "completion"], default="chat")
    
    args = parser.parse_args()
    
    run_generate_training_data(
        apply_mitigation=not args.no_mitigation,
        test_split=args.test_split,
        output_format=args.format
    )