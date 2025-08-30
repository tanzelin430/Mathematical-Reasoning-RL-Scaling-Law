#!/usr/bin/env python3
"""
Sample math data based on difficulty levels (pass rate).
Divides data into 2 difficulty levels based on qwen2.5_7b_pass_rate:
- Easy: pass_rate >= 0.3
- Hard: pass_rate < 0.3
Samples equal numbers from each difficulty level (1:1 ratio).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Any, Dict, List

def unify_math_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    """Apply VeRL format to math data (same as pre_verl.py)"""
    prompt = row.get("prompt", [])
    
    # Ensure prompt ends with the boxed instruction if not already present
    if prompt and len(prompt) > 0:
        prompt_list = prompt.tolist() if hasattr(prompt, 'tolist') else prompt
        if prompt_list and isinstance(prompt_list[0], dict) and prompt_list[0].get('role') == 'user':
            content = prompt_list[0].get('content', '')
            # Add boxed instruction if not present
            if '\\boxed{' not in content:
                prompt_list[0]['content'] = content + " Please output the final answer within \\boxed{}."
                prompt = np.array(prompt_list) if hasattr(prompt, 'tolist') else prompt_list
    
    # Include pass rates in extra_info
    extra_info = {"split": split, "index": idx}
    
    # Add pass rates if available
    if pd.notna(row.get("qwen2.5_7b_pass_rate")):
        extra_info["qwen2.5_7b_pass_rate"] = float(row["qwen2.5_7b_pass_rate"])
    if pd.notna(row.get("qwen3_30b_pass_rate")):
        extra_info["qwen3_30b_pass_rate"] = float(row["qwen3_30b_pass_rate"])
    
    return {
        "data_source": row.get("data_source", "math_unknown"),
        "prompt": prompt,
        "ability": row.get("ability", "math"),
        "reward_model": row.get("reward_model", {}),
        "extra_info": extra_info
    }

def categorize_difficulty(pass_rate):
    """Categorize samples into difficulty levels based on pass rate"""
    if pd.isna(pass_rate):
        # If pass rate is missing, assign to hard difficulty (conservative approach)
        return "hard"
    elif pass_rate >= 0.3:
        return "easy"
    else:
        return "hard"

def main():
    parser = argparse.ArgumentParser(description='Sample math data by difficulty levels')
    parser.add_argument('--total_samples', type=int, required=True, help='Total number of samples')
    parser.add_argument('--output_dir', type=str, default='../../data/difficulty_balanced_math', help='Output directory for sampled data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()
    
    # Calculate samples per difficulty level
    samples_per_difficulty = args.total_samples // 2
    remainder = args.total_samples % 2
    
    print(f"Total samples requested: {args.total_samples}")
    print(f"Base samples per difficulty level: {samples_per_difficulty}")
    if remainder:
        print(f"Remainder {remainder} will be added to the hard difficulty level")
    
    # Read original math data with pass rates
    base_dir = Path("/fs-computility/mabasic/shared/data/guru-RL-92k/train")
    math_file = base_dir / "math__combined_54.4k.parquet"
    
    print(f"\nReading data from: {math_file}")
    df = pd.read_parquet(math_file)
    print(f"Total samples available: {len(df)}")
    
    # Categorize by difficulty
    df['difficulty'] = df['qwen2.5_7b_pass_rate'].apply(categorize_difficulty)
    
    # Check distribution
    print("\nDifficulty distribution:")
    difficulty_counts = df['difficulty'].value_counts()
    for diff in ['easy', 'hard']:
        count = difficulty_counts.get(diff, 0)
        if diff == 'easy':
            print(f"  {diff} (pass_rate >= 0.3): {count} samples")
        else:
            print(f"  {diff} (pass_rate < 0.3): {count} samples")
    
    # Sample from each difficulty level
    sampled_dfs = []
    difficulty_levels = ['easy', 'hard']
    
    for i, difficulty in enumerate(difficulty_levels):
        # Add remainder to hard difficulty
        target_samples = samples_per_difficulty + (remainder if difficulty == 'hard' else 0)
        
        # Get samples for this difficulty
        difficulty_df = df[df['difficulty'] == difficulty]
        
        if len(difficulty_df) < target_samples:
            print(f"\nWARNING: {difficulty} has only {len(difficulty_df)} samples, but {target_samples} requested.")
            print(f"Taking all {len(difficulty_df)} samples from {difficulty} level.")
            sampled = difficulty_df
        else:
            sampled = difficulty_df.sample(n=target_samples, random_state=args.seed + i)
        
        sampled_dfs.append(sampled)
        print(f"\nSampled {len(sampled)} from {difficulty} level")
        
        # Show some statistics
        pass_rates = sampled['qwen2.5_7b_pass_rate'].dropna()
        if len(pass_rates) > 0:
            print(f"  Pass rate stats: min={pass_rates.min():.3f}, max={pass_rates.max():.3f}, mean={pass_rates.mean():.3f}")
    
    # Combine all sampled data
    final_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Sort by pass rate in descending order (highest/easiest first, lowest/hardest last)
    # Use 0.0 as default for NaN values (treat them as hardest)
    final_df['sort_key'] = final_df['qwen2.5_7b_pass_rate'].fillna(0.0)
    final_df = final_df.sort_values('sort_key', ascending=False).reset_index(drop=True)
    final_df = final_df.drop('sort_key', axis=1)
    
    print(f"\nTotal samples selected: {len(final_df)}")
    
    # Convert to VeRL format
    print("\nConverting to VeRL format...")
    unified_data = []
    for idx, row in final_df.iterrows():
        rec = unify_math_data(row, idx, "train")
        unified_data.append(rec)
    
    # Save the output
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"math__difficulty_balanced_{len(final_df)}.parquet"
    pd.DataFrame(unified_data).to_parquet(output_file, engine='pyarrow')
    
    print(f"\nSaved to: {output_file}")
    
    # Print training file list format for easy copying
    print("\nFor training scripts, use:")
    abs_path = output_file.resolve()
    print(f'data.train_files="[\'{abs_path}\']"')

if __name__ == '__main__':
    main()