#!/usr/bin/env python3
"""
Prepare full math dataset ordered by difficulty (easy -> hard) and converted to VeRL format.
- Uses the entire math__combined_54.4k.parquet (no sampling)
- Difficulty based on qwen2.5_7b_pass_rate (>= 0.3 is easy, otherwise hard)
- Sorted by pass rate descending (NaN treated as hardest with 0.0)
- Splits out a difficulty-balanced test set (default size 500) and uses the rest for training
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def unify_math_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    """Apply VeRL-like format to math data (aligned with sample_math_by_difficulty.py)."""
    prompt = row.get("prompt", [])

    # Ensure prompt ends with the boxed instruction if not already present
    if prompt and len(prompt) > 0:
        prompt_list = prompt.tolist() if hasattr(prompt, "tolist") else prompt
        if (
            prompt_list
            and isinstance(prompt_list[0], dict)
            and prompt_list[0].get("role") == "user"
        ):
            content = prompt_list[0].get("content", "")
            if "\\boxed{" not in content:
                prompt_list[0][
                    "content"
                ] = content + " Please output the final answer within \\boxed{}."
                prompt = np.array(prompt_list) if hasattr(prompt, "tolist") else prompt_list

    extra_info: Dict[str, Any] = {"split": split, "index": idx}

    if pd.notna(row.get("qwen2.5_7b_pass_rate")):
        extra_info["qwen2.5_7b_pass_rate"] = float(row["qwen2.5_7b_pass_rate"])  # type: ignore[index]
    if pd.notna(row.get("qwen3_30b_pass_rate")):
        extra_info["qwen3_30b_pass_rate"] = float(row["qwen3_30b_pass_rate"])  # type: ignore[index]

    return {
        "data_source": row.get("data_source", "math_unknown"),
        "prompt": prompt,
        "ability": row.get("ability", "math"),
        "reward_model": row.get("reward_model", {}),
        "extra_info": extra_info,
    }


def categorize_difficulty(pass_rate: Any) -> str:
    if pd.isna(pass_rate):
        return "hard"
    return "easy" if float(pass_rate) >= 0.3 else "hard"


def stratified_test_split(
    df: pd.DataFrame, test_size: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a difficulty-balanced test set from df.

    Strategy:
    - Target half easy, half hard
    - If a bucket lacks enough samples, fill remainder from the other bucket
    - Deterministic sampling controlled by seed
    """
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be > 0 and smaller than dataset size")

    easy_df = df[df["difficulty"] == "easy"]
    hard_df = df[df["difficulty"] == "hard"]

    desired_easy = test_size // 2
    desired_hard = test_size - desired_easy

    easy_available = len(easy_df)
    hard_available = len(hard_df)

    easy_take = min(desired_easy, easy_available)
    hard_take = min(desired_hard, hard_available)

    remaining = test_size - (easy_take + hard_take)

    if remaining > 0:
        # try fill from easy first
        extra_easy_capacity = max(0, easy_available - easy_take)
        extra_from_easy = min(remaining, extra_easy_capacity)
        easy_take += extra_from_easy
        remaining -= extra_from_easy

        if remaining > 0:
            extra_hard_capacity = max(0, hard_available - hard_take)
            extra_from_hard = min(remaining, extra_hard_capacity)
            hard_take += extra_from_hard
            remaining -= extra_from_hard

    # Sample
    sampled_easy = easy_df.sample(n=easy_take, random_state=seed) if easy_take > 0 else easy_df.iloc[0:0]
    sampled_hard = hard_df.sample(n=hard_take, random_state=seed + 1) if hard_take > 0 else hard_df.iloc[0:0]

    test_df = pd.concat([sampled_easy, sampled_hard], axis=0)
    train_df = df.drop(index=test_df.index)

    # Order both by pass rate desc (easy->hard)
    for part in (test_df, train_df):
        part["sort_key"] = part["qwen2.5_7b_pass_rate"].fillna(0.0).astype(float)
        part.sort_values("sort_key", ascending=False, inplace=True)
        part.drop(columns=["sort_key"], inplace=True)

    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare full math dataset ordered by difficulty and converted to VeRL format",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/mnt/shared-storage-user/ma4agi-gpu/data/dataset/guru-RL-92k/train/math__combined_54.4k.parquet",
        help="Input parquet containing full math data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../data/math_full_by_difficulty",
        help="Directory to write the processed parquet",
    )
    parser.add_argument(
        "--pass_rate_col",
        type=str,
        default="qwen2.5_7b_pass_rate",
        help="Column name containing pass rate used for difficulty and sorting",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=500,
        help="Number of samples to hold out for a difficulty-balanced test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading full dataset: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Total rows: {len(df)}")

    if args.pass_rate_col not in df.columns:
        raise ValueError(
            f"Pass rate column '{args.pass_rate_col}' not found in input file. Available columns: {list(df.columns)}"
        )

    # Categorize and sort
    df["difficulty"] = df[args.pass_rate_col].apply(categorize_difficulty)
    print("Difficulty value counts:")
    print(df["difficulty"].value_counts(dropna=False))

    # Sort easy->hard: descending pass rate, NaN treated as 0.0 (hardest)
    df["sort_key"] = df[args.pass_rate_col].fillna(0.0).astype(float)
    df = df.sort_values("sort_key", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["sort_key"])  # keep dataset clean

    # Split test set (balanced by difficulty)
    print(f"\nCreating difficulty-balanced test split: size={args.test_size}, seed={args.seed}")
    train_df, test_df = stratified_test_split(df, args.test_size, args.seed)
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    print("Test difficulty counts:")
    print(test_df["difficulty"].value_counts(dropna=False))

    # Convert to VeRL format
    print("\nConverting TRAIN to VeRL format...")
    unified_train: List[Dict[str, Any]] = []
    for idx, row in train_df.iterrows():
        unified_train.append(unify_math_data(row, idx, "train"))
        if (idx + 1) % 10000 == 0:
            print(f"  Converted {idx + 1} train rows...")

    print("Converting TEST to VeRL format...")
    unified_test: List[Dict[str, Any]] = []
    for idx, row in test_df.iterrows():
        unified_test.append(unify_math_data(row, idx, "test"))

    # Write outputs
    train_file = output_dir / f"math__full_difficulty_ordered_train_{len(unified_train)}.parquet"
    test_file = output_dir / f"math__full_difficulty_ordered_test_{len(unified_test)}.parquet"

    pd.DataFrame(unified_train).to_parquet(train_file, engine="pyarrow")
    pd.DataFrame(unified_test).to_parquet(test_file, engine="pyarrow")

    print(f"\nSaved TRAIN to: {train_file}")
    print(f"Saved TEST  to: {test_file}")

    abs_train = train_file.resolve()
    abs_test = test_file.resolve()
    print("\nFor training usage:")
    print(f"data.train_files=\"['{abs_train}']\"")
    print(f"data.eval_files=\"['{abs_test}']\"")


if __name__ == "__main__":
    main() 