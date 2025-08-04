#!/usr/bin/env python3
"""
Preprocess guru-RL-92k data to unify format across domains
"""
import pandas as pd
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def unify_math_data(df):
    """Unify math data format"""
    # Rename solution to ground_truth
    if 'solution' in df.columns and 'ground_truth' not in df.columns:
        df['ground_truth'] = df['solution']
    
    # Ensure domain field
    if 'domain' not in df.columns:
        df['domain'] = 'math'
    
    return df

def unify_code_data(df):
    """Unify code data format"""
    # Create ground_truth from test cases or expected output
    if 'ground_truth' not in df.columns:
        if 'test' in df.columns:
            # Extract expected output from test cases
            df['ground_truth'] = df['test'].apply(lambda x: str(x) if x else '')
        elif 'input_output' in df.columns:
            # Use input_output as ground truth
            df['ground_truth'] = df['input_output'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
            )
        else:
            df['ground_truth'] = ''
    
    # Add domain
    if 'domain' not in df.columns:
        df['domain'] = 'code'
        
    return df

def unify_logic_data(df):
    """Logic data already has ground_truth, just ensure consistency"""
    # Add domain if missing
    if 'domain' not in df.columns:
        df['domain'] = 'logic'
        
    # Ensure ground_truth is string
    if 'ground_truth' in df.columns:
        def safe_stringify(x):
            try:
                if isinstance(x, dict):
                    return json.dumps(x)
                elif hasattr(x, 'tolist'):  # numpy array
                    return json.dumps(x.tolist())
                else:
                    return str(x)
            except:
                return str(x)
        
        df['ground_truth'] = df['ground_truth'].apply(safe_stringify)
    
    return df

def process_file(input_file, output_file):
    """Process a single file"""
    logger.info(f"Processing {input_file}...")
    
    df = pd.read_parquet(input_file)
    original_len = len(df)
    
    # Determine domain from filename
    if 'math' in str(input_file):
        df = unify_math_data(df)
    elif 'codegen' in str(input_file) or 'code' in str(input_file):
        df = unify_code_data(df)
    elif 'logic' in str(input_file):
        df = unify_logic_data(df)
    else:
        logger.warning(f"Unknown domain for {input_file}")
    
    # Ensure required columns exist
    required_columns = ['prompt', 'ground_truth', 'data_source']
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"Missing column {col} in {input_file}")
            if col == 'prompt' and 'query' in df.columns:
                df['prompt'] = df['query']
            elif col == 'prompt' and 'instruction' in df.columns:
                df['prompt'] = df['instruction']
            else:
                df[col] = ''
    
    # Save processed file
    df.to_parquet(output_file)
    logger.info(f"Saved {len(df)} samples to {output_file}")
    
    return df

def main():
    """Process all guru-RL-92k files"""
    base_dir = Path("/fs-computility/mabasic/shared/data/guru-RL-92k")
    output_dir = Path("../../data/guru_unified")
    
    # Create output directories
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "online_eval").mkdir(parents=True, exist_ok=True)
    
    # Process training files
    train_files = [
        "math__combined_54.4k.parquet",
        "codegen__leetcode2k_1.3k.parquet",
        "codegen__livecodebench_440.parquet",
        "codegen__primeintellect_7.5k.parquet",
        "codegen__taco_8.8k.parquet",
        "logic__arcagi1_111.parquet",
        "logic__arcagi2_190.parquet",
        "logic__barc_1.6k.parquet",
        "logic__graph_logical_1.2k.parquet",
        "logic__ordering_puzzle_1.9k.parquet",
        "logic__zebra_puzzle_1.3k.parquet"
    ]
    
    for file in train_files:
        input_path = base_dir / "train" / file
        output_path = output_dir / "train" / file
        
        if input_path.exists():
            process_file(input_path, output_path)
        else:
            logger.warning(f"File not found: {input_path}")
    
    # Process validation files
    val_files = [
        "math__aime_repeated_8x_240.parquet",
        "math__amc_repeated_4x_332.parquet",
        "math__math_500.parquet",
        "codegen__humaneval_164.parquet",
        "codegen__livecodebench_279.parquet",
        "codegen__mbpp_200.parquet",
        "logic__ordering_puzzle_dataset_100.parquet",
        "logic__zebra_puzzle_dataset_200.parquet"
    ]
    
    for file in val_files:
        input_path = base_dir / "online_eval" / file
        output_path = output_dir / "online_eval" / file
        
        if input_path.exists():
            process_file(input_path, output_path)
        else:
            logger.warning(f"File not found: {input_path}")
    
    logger.info("Data preprocessing complete!")
    
    # Print file lists for config
    print("\nProcessed training files:")
    train_processed = list((output_dir / "train").glob("*.parquet"))
    print(json.dumps([str(f) for f in sorted(train_processed)], indent=2))
    
    print("\nProcessed validation files:")
    val_processed = list((output_dir / "online_eval").glob("*.parquet"))
    print(json.dumps([str(f) for f in sorted(val_processed)], indent=2))

if __name__ == "__main__":
    main()