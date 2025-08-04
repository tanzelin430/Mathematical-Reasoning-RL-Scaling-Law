#!/usr/bin/env python3
"""
Preprocessing script for guru-RL-92k data to VeRL format.
Preserves the original message format with roles.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Any, Dict, List, Union


def extract_messages(row: pd.Series) -> List[Dict[str, str]]:
    """Extract messages in the format expected by VeRL."""
    # Check if prompt field exists and is in the right format
    if 'prompt' in row and isinstance(row['prompt'], np.ndarray):
        # Original format: array of message dicts
        messages = row['prompt'].tolist()
        if messages and isinstance(messages[0], dict) and 'role' in messages[0]:
            return messages
    
    # Fallback: create a simple user message
    prompt_text = ""
    for col in ['prompt', 'query', 'problem', 'question']:
        if col in row and pd.notna(row[col]):
            if isinstance(row[col], str):
                prompt_text = row[col]
                break
            elif isinstance(row[col], np.ndarray) and row[col].size > 0:
                # Extract text from array
                first_elem = row[col][0] if row[col].size == 1 else row[col]
                if isinstance(first_elem, dict) and 'content' in first_elem:
                    prompt_text = first_elem['content']
                else:
                    prompt_text = str(first_elem)
                break
    
    if prompt_text:
        return [{"role": "user", "content": prompt_text}]
    return []


def get_answer(row: pd.Series) -> str:
    """Extract answer/solution from various columns."""
    for col in ['solution', 'response', 'completion', 'answer']:
        if col in row and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, str):
                return val
            elif isinstance(val, np.ndarray):
                # Skip numpy arrays
                continue
            else:
                return str(val)
    return ""


def get_ground_truth(row: pd.Series, domain: str) -> str:
    """Extract ground truth for evaluation."""
    # Check reward_model dict first
    if 'reward_model' in row and isinstance(row['reward_model'], dict):
        if 'ground_truth' in row['reward_model']:
            return str(row['reward_model']['ground_truth'])
    
    # Check direct ground_truth column
    if 'ground_truth' in row:
        value = row['ground_truth']
        if value is not None and not pd.isna(value) and not isinstance(value, np.ndarray):
            return str(value)
    
    # For code domain, might use test cases
    if domain == 'code' and 'test' in row:
        value = row['test']
        if value is not None and not pd.isna(value) and not isinstance(value, np.ndarray):
            return str(value)
    
    return ""


def process_file(input_path: Path, output_path: Path, domain: str) -> bool:
    """Process a single parquet file to VeRL format."""
    print(f"Processing {input_path.name}...")
    
    try:
        # Read parquet file
        df = pd.read_parquet(input_path)
        print(f"  Original shape: {df.shape}")
        
        # Create standardized dataframe
        standardized_data = []
        
        for idx, row in df.iterrows():
            try:
                # Extract messages
                messages = extract_messages(row)
                if not messages:
                    continue
                
                # Get answer
                answer = get_answer(row)
                if not answer:
                    continue
                
                # Get reward_model info
                reward_model = row.get('reward_model', {})
                if not isinstance(reward_model, dict):
                    ground_truth = get_ground_truth(row, domain)
                    reward_model = {'ground_truth': ground_truth, 'style': 'rule'}
                
                # Check if apply_chat_template should be used
                apply_chat_template = row.get('apply_chat_template', True)
                if isinstance(apply_chat_template, (np.bool_, bool)):
                    apply_chat_template = bool(apply_chat_template)
                else:
                    apply_chat_template = True
                
                std_row = {
                    'prompt': messages,  # Keep as list of dicts with roles
                    'answer': answer,
                    'ground_truth': get_ground_truth(row, domain),
                    'domain': domain,
                    'data_source': row.get('data_source', f'{domain}_unknown'),
                    'reward_model': reward_model,
                    'apply_chat_template': apply_chat_template,
                    'extra_info': {
                        'index': idx,
                        'original_index': row.get('extra_info', {}).get('index', idx) if isinstance(row.get('extra_info', {}), dict) else idx
                    }
                }
                
                standardized_data.append(std_row)
                    
            except Exception as e:
                print(f"  Warning: Skipping row {idx}: {type(e).__name__}: {e}")
                continue
        
        if not standardized_data:
            print(f"  ERROR: No valid data extracted from {input_path}")
            return False
            
        # Create new dataframe
        new_df = pd.DataFrame(standardized_data)
        
        # Save to parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(output_path, engine='pyarrow')
        
        print(f"  Saved to {output_path}")
        print(f"  New shape: {new_df.shape}")
        return True
        
    except Exception as e:
        print(f"  ERROR processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Define directories
    base_dir = Path("/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law")
    input_dir = base_dir / "data" / "guru_unified"
    output_dir = base_dir / "data" / "guru_verl"
    
    # Process selected files for testing
    files_to_process = [
        # Math
        ("train/math__combined_54.4k.parquet", "math"),
        # Code
        ("train/codegen__leetcode2k_1.3k.parquet", "code"),
        # Logic
        ("train/logic__zebra_puzzle_1.3k.parquet", "logic"),
    ]
    
    success_count = 0
    for rel_path, domain in files_to_process:
        input_path = input_dir / rel_path
        output_path = output_dir / rel_path
        
        if input_path.exists():
            if process_file(input_path, output_path, domain):
                success_count += 1
        else:
            print(f"File not found: {input_path}")
    
    print(f"\nProcessing complete! Successfully processed {success_count}/{len(files_to_process)} files")
    print(f"VeRL-formatted data saved to: {output_dir}")


if __name__ == "__main__":
    main()