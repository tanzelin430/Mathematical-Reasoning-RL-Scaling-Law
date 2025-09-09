#!/usr/bin/env python3
"""
Sample dataset by difficulty distribution while preserving original proportions.

This script takes a difficulty-ordered dataset and creates N smaller subsets,
each with 1/N of the original size while maintaining the same difficulty distribution.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def categorize_difficulty(pass_rate: Any) -> str:
    """Categorize difficulty based on pass rate (same logic as prepare_math_by_dificulty_full.py)."""
    if pd.isna(pass_rate):
        return "hard"
    return "easy" if float(pass_rate) >= 0.3 else "hard"


def stratified_sample(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    """
    Sample from df while preserving difficulty distribution.
    
    Args:
        df: Input dataframe with difficulty column
        target_size: Target number of samples
        seed: Random seed for reproducibility
    
    Returns:
        Sampled dataframe maintaining original difficulty proportions
    """
    if target_size >= len(df):
        print(f"Warning: Target size {target_size} >= dataset size {len(df)}, returning full dataset")
        return df.copy()
    
    # Calculate original proportions
    difficulty_counts = df["difficulty"].value_counts()
    total_samples = len(df)
    
    easy_proportion = difficulty_counts.get("easy", 0) / total_samples
    hard_proportion = difficulty_counts.get("hard", 0) / total_samples
    
    print(f"Original difficulty distribution:")
    print(f"  Easy: {difficulty_counts.get('easy', 0)} ({easy_proportion:.3f})")
    print(f"  Hard: {difficulty_counts.get('hard', 0)} ({hard_proportion:.3f})")
    
    # Calculate target counts
    target_easy = int(target_size * easy_proportion)
    target_hard = target_size - target_easy
    
    # Get available samples by difficulty
    easy_df = df[df["difficulty"] == "easy"]
    hard_df = df[df["difficulty"] == "hard"]
    
    easy_available = len(easy_df)
    hard_available = len(hard_df)
    
    # Adjust if not enough samples in a category
    actual_easy = min(target_easy, easy_available)
    actual_hard = min(target_hard, hard_available)
    
    # Fill remaining from the other category if needed
    remaining = target_size - (actual_easy + actual_hard)
    if remaining > 0:
        extra_easy_capacity = max(0, easy_available - actual_easy)
        extra_from_easy = min(remaining, extra_easy_capacity)
        actual_easy += extra_from_easy
        remaining -= extra_from_easy
        
        if remaining > 0:
            extra_hard_capacity = max(0, hard_available - actual_hard)
            actual_hard += min(remaining, extra_hard_capacity)
    
    print(f"Target difficulty distribution for {target_size} samples:")
    print(f"  Easy: {actual_easy} ({actual_easy/target_size:.3f})")
    print(f"  Hard: {actual_hard} ({actual_hard/target_size:.3f})")
    
    # Sample from each difficulty group
    sampled_parts = []
    if actual_easy > 0:
        sampled_easy = easy_df.sample(n=actual_easy, random_state=seed)
        sampled_parts.append(sampled_easy)
    if actual_hard > 0:
        sampled_hard = hard_df.sample(n=actual_hard, random_state=seed + 1)
        sampled_parts.append(sampled_hard)
    
    # Combine and maintain original ordering (easy -> hard)
    if sampled_parts:
        result = pd.concat(sampled_parts, axis=0)
        # Sort by original index to maintain easy->hard ordering
        result = result.sort_index()
        return result
    else:
        return df.iloc[0:0]  # Return empty dataframe with same structure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample dataset by difficulty distribution while preserving proportions"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/math_curriculum/math__full_difficulty_ordered_train_53904.parquet",
        help="Input parquet file with difficulty-ordered data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="data/math_curriculum_sampled",
        help="Directory to save sampled datasets"
    )
    parser.add_argument(
        "N",
        type=int,
        help="Number of splits (final size will be original_size / N)"
    )
    parser.add_argument(
        "--pass_rate_col",
        type=str,
        default="qwen2.5_7b_pass_rate",
        help="Pass rate column name for difficulty categorization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )
    args = parser.parse_args()

    if args.N <= 0:
        raise ValueError("N must be positive")

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {input_path}")
    df = pd.read_parquet(input_path)
    original_size = len(df)
    target_size = original_size // args.N
    
    print(f"Original dataset size: {original_size}")
    print(f"Target size (1/{args.N}): {target_size}")
    
    if target_size == 0:
        raise ValueError(f"Dataset too small for {args.N} splits")

    # Extract difficulty information from extra_info
    print("Extracting difficulty information...")
    pass_rates = []
    for _, row in df.iterrows():
        extra_info = row["extra_info"]
        if isinstance(extra_info, dict) and args.pass_rate_col in extra_info:
            pass_rates.append(extra_info[args.pass_rate_col])
        else:
            pass_rates.append(np.nan)
    
    df["pass_rate"] = pass_rates
    df["difficulty"] = df["pass_rate"].apply(categorize_difficulty)
    
    print("\nOriginal difficulty distribution:")
    print(df["difficulty"].value_counts())
    
    # Perform stratified sampling
    print(f"\nSampling {target_size} samples with preserved difficulty distribution...")
    sampled_df = stratified_sample(df, target_size, args.seed)
    
    # Remove temporary columns before saving
    sampled_df = sampled_df.drop(columns=["pass_rate", "difficulty"])
    
    # Save result
    output_file = output_dir / f"math__sampled_1_{args.N}_{len(sampled_df)}.parquet"
    sampled_df.to_parquet(output_file, engine="pyarrow")
    
    print(f"\nSaved sampled dataset to: {output_file}")
    print(f"Final size: {len(sampled_df)} samples")
    
    # Print usage example
    abs_path = output_file.resolve()
    print(f"\nFor training usage:")
    print(f'data.train_files="[\'${abs_path}\']"')


if __name__ == "__main__":
    main()