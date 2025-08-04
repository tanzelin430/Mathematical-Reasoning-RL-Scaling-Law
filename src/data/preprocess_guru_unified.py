#!/usr/bin/env python3
"""
Improved preprocessing script for guru-RL-92k data to create unified format.
Handles numpy arrays, nested data structures, and column standardization.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, date
import re


def serialize_value(value: Any) -> Any:
    """Safely serialize values including numpy arrays and nested structures."""
    if isinstance(value, np.ndarray):
        # Convert numpy array to list, handling array of dicts
        if value.dtype == object:
            # Array of objects (like dicts)
            return [serialize_value(item) for item in value.tolist()]
        else:
            return value.tolist()
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [serialize_value(item) for item in value]
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif pd.isna(value):
        return None
    else:
        return value


def format_prompt(prompt_data: Any) -> str:
    """Convert various prompt formats to a consistent string format."""
    if isinstance(prompt_data, str):
        return prompt_data
    elif isinstance(prompt_data, list):
        # Handle list of message dicts
        messages = []
        for msg in prompt_data:
            if isinstance(msg, dict) and 'content' in msg:
                messages.append(msg['content'])
            else:
                messages.append(str(msg))
        return "\n".join(messages)
    elif isinstance(prompt_data, np.ndarray):
        # Handle numpy array of messages
        return format_prompt(prompt_data.tolist())
    elif isinstance(prompt_data, dict) and 'content' in prompt_data:
        return prompt_data['content']
    else:
        return str(prompt_data)


def get_answer_from_data(row: pd.Series, domain: str) -> str:
    """Extract the answer/solution from various column names based on domain."""
    # Priority order for answer columns
    answer_columns = ['solution', 'response', 'completion', 'answer', 'ground_truth']
    
    for col in answer_columns:
        if col in row:
            value = row[col]
            # Handle numpy arrays
            if isinstance(value, np.ndarray):
                continue  # Skip numpy arrays for answer extraction
            # Check if value is valid
            if value is not None and not pd.isna(value):
                str_value = str(value).strip()
                if str_value:
                    return str_value
    
    # If no answer found, return empty string
    return ""


def get_ground_truth_from_data(row: pd.Series, domain: str) -> str:
    """Extract ground truth from various sources."""
    # Check reward_model dict first
    if 'reward_model' in row and isinstance(row['reward_model'], dict):
        if 'ground_truth' in row['reward_model']:
            return str(row['reward_model']['ground_truth'])
    
    # Check direct ground_truth column
    if 'ground_truth' in row:
        value = row['ground_truth']
        # Skip numpy arrays
        if isinstance(value, np.ndarray):
            pass
        elif value is not None and not pd.isna(value):
            # For math domain, ground_truth might be the full solution
            if domain == 'math' and len(str(value)) > 100:
                # Extract answer from the solution if possible
                solution = str(value)
                if '\\boxed{' in solution:
                    matches = re.findall(r'\\boxed\{([^}]+)\}', solution)
                    if matches:
                        return matches[-1]
            return str(value)
    
    # For code domain, might use test cases
    if domain == 'code' and 'test' in row:
        value = row['test']
        if not isinstance(value, np.ndarray) and value is not None and not pd.isna(value):
            return str(value)
    
    # Default to answer if no ground truth found
    return get_answer_from_data(row, domain)


def standardize_row(row: pd.Series, domain: str) -> Dict[str, Any]:
    """Standardize a single row to unified format."""
    # Extract prompt
    prompt = ""
    if 'prompt' in row:
        prompt = format_prompt(row['prompt'])
    elif 'query' in row:
        prompt = str(row['query'])
    elif 'problem' in row:
        prompt = str(row['problem'])
    
    # Extract answer
    answer = get_answer_from_data(row, domain)
    
    # Extract ground truth
    ground_truth = get_ground_truth_from_data(row, domain)
    
    # Extract extra info
    extra_info = {}
    if 'extra_info' in row:
        extra_info = serialize_value(row['extra_info'])
    
    # Add other relevant fields to extra_info
    for col in ['task_id', 'entry_point', 'test', 'input_output', 'meta']:
        if col in row and pd.notna(row[col]):
            extra_info[col] = serialize_value(row[col])
    
    # Create standardized row
    standardized = {
        'prompt': prompt,
        'answer': answer,
        'ground_truth': ground_truth,
        'domain': domain,
        'data_source': row.get('data_source', f'{domain}_unknown'),
        'extra_info': json.dumps(extra_info),  # Store as JSON string
    }
    
    return standardized


def process_file(input_path: Path, output_path: Path, domain: str) -> None:
    """Process a single parquet file to standardized format."""
    print(f"Processing {input_path.name}...")
    
    try:
        # Read parquet file
        df = pd.read_parquet(input_path)
        print(f"  Original shape: {df.shape}")
        print(f"  Original columns: {df.columns.tolist()}")
        
        # Standardize each row
        standardized_rows = []
        for idx, row in df.iterrows():
            try:
                std_row = standardize_row(row, domain)
                standardized_rows.append(std_row)
            except Exception as e:
                print(f"  Warning: Failed to process row {idx}: {e}")
                continue
        
        # Create new dataframe
        new_df = pd.DataFrame(standardized_rows)
        
        # Ensure all columns are present
        required_columns = ['prompt', 'answer', 'ground_truth', 'domain', 'data_source', 'extra_info']
        for col in required_columns:
            if col not in new_df.columns:
                new_df[col] = ""
        
        # Save to parquet with simple schema
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define schema explicitly to avoid Arrow issues
        schema = pa.schema([
            ('prompt', pa.string()),
            ('answer', pa.string()),
            ('ground_truth', pa.string()),
            ('domain', pa.string()),
            ('data_source', pa.string()),
            ('extra_info', pa.string()),
        ])
        
        table = pa.Table.from_pandas(new_df[required_columns], schema=schema)
        pq.write_table(table, output_path)
        
        print(f"  Saved to {output_path}")
        print(f"  New shape: {new_df.shape}")
        
    except Exception as e:
        print(f"  ERROR processing {input_path}: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Define input and output directories
    base_dir = Path("/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law")
    input_dir = base_dir / "data" / "guru_unified"
    output_dir = base_dir / "data" / "guru_standardized"
    
    # Process train files
    train_files = [
        ("math__combined_54.4k.parquet", "math"),
        ("codegen__leetcode2k_1.3k.parquet", "code"),
        ("codegen__livecodebench_440.parquet", "code"),
        ("codegen__primeintellect_7.5k.parquet", "code"),
        ("codegen__taco_8.8k.parquet", "code"),
        ("logic__arcagi1_111.parquet", "logic"),
        ("logic__arcagi2_190.parquet", "logic"),
        ("logic__barc_1.6k.parquet", "logic"),
        ("logic__graph_logical_1.2k.parquet", "logic"),
        ("logic__ordering_puzzle_1.9k.parquet", "logic"),
        ("logic__zebra_puzzle_1.3k.parquet", "logic"),
    ]
    
    print("Processing training files...")
    for filename, domain in train_files:
        input_path = input_dir / "train" / filename
        output_path = output_dir / "train" / filename
        if input_path.exists():
            process_file(input_path, output_path, domain)
    
    # Process eval files
    eval_files = [
        ("math__math_500.parquet", "math"),
        ("math__aime_repeated_8x_240.parquet", "math"),
        ("math__amc_repeated_4x_332.parquet", "math"),
        ("codegen__humaneval_164.parquet", "code"),
    ]
    
    print("\nProcessing evaluation files...")
    for filename, domain in eval_files:
        input_path = input_dir / "online_eval" / filename
        output_path = output_dir / "online_eval" / filename
        if input_path.exists():
            process_file(input_path, output_path, domain)
    
    print("\nPreprocessing complete!")
    print(f"Standardized data saved to: {output_dir}")


if __name__ == "__main__":
    main()