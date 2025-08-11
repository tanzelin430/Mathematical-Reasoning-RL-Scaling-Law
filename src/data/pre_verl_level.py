#!/usr/bin/env python3
"""
Convert source parquet rows into VeRL format and split into three difficulty
levels (easy / medium / hard) based on configurable pass-rate thresholds for
two model columns (e.g., 7B and 30B).

Each output example is a dict with keys: data_source, prompt, ability,
reward_model, extra_info. The extra_info keeps its original format.
"""
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def unify_math_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    return {
            "data_source": row.get("data_source", "math_unknown"),
            "prompt": row.get("prompt", []),
            "ability": row.get("ability", "math"),
            "reward_model": row.get("reward_model", {}),
            "extra_info": {"split": split, "index": idx}
        }


def unify_code_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    question = ''
    if pd.notna(row.get('prompt')):
        question = str(row['prompt'])
    elif 'query' in row and pd.notna(row['query']):
        question = str(row['query'])

    # Extract test data from reward_model field (Reasoning360 format)
    solution = ''
    if isinstance(row.get('reward_model'), dict):
        # Get ground_truth from reward_model dict
        ground_truth = row['reward_model'].get('ground_truth', '')
        if ground_truth:
            # The ground_truth is already a JSON string with test data
            # For leetcode: {"functional": "def check(candidate)..."}
            # For others: {"inputs": [...], "outputs": [...]}
            # We'll pass it through as-is since coder1 expects JSON
            solution = ground_truth
    elif pd.notna(row.get('ground_truth')):
        # Fallback to old format if exists
        solution = str(row['ground_truth'])
    elif pd.notna(row.get('test')):
        # Fallback to test field if exists
        solution = str(row['test'])

    data_source = row.get('data_source', 'code_unknown')

    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": row.get("ability", "codegen"),  # Use 'codegen' to match the data
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx}
    }
    return data


def unify_logic_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    solution = ''
    if isinstance(row.get('reward_model'), dict):
        solution = str(row['reward_model'].get('ground_truth', ''))
    elif pd.notna(row.get('ground_truth')):
        solution = str(row['ground_truth'])

    data_source = row.get('data_source', 'logic_unknown')

    data = {
        "data_source": data_source,
        "prompt": row.get("prompt", []),
        "ability": row.get("ability", "logic"),
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx}
    }
    return data


def unify_stem_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    solution = ''
    if isinstance(row.get('reward_model'), dict):
        solution = str(row['reward_model'].get('ground_truth', ''))

    data_source = row.get('data_source', 'stem_unknown')

    data = {
        "data_source": data_source,
        "prompt": row.get("prompt", []),
        "ability": "stem",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx}
    }
    return data


def unify_simulation_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    data_source = row.get('data_source', 'simulation_unknown')
    
    data = {
        "data_source": data_source,
        "prompt": row.get("prompt", []),
        "ability": row.get("ability", "coding-inference"),
        "reward_model": row.get("reward_model", {}),
        "extra_info": {"split": split, "index": idx}
    }
    return data


def _parse_threshold_pair(pair_str: str) -> Tuple[float, float]:
    parts = [p.strip() for p in str(pair_str).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Threshold must be 'low,high', got: {pair_str}")
    low, high = float(parts[0]), float(parts[1])
    if not (0.0 <= low <= 1.0 and 0.0 <= high <= 1.0 and low < high):
        raise ValueError(f"Invalid thresholds: low={low}, high={high}")
    return low, high


def _safe_get_float(row: pd.Series, key: Optional[str]) -> Optional[float]:
    if not key:
        return None
    if key not in row:
        return None
    val = row.get(key)
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _classify_difficulty(pass_rate: Optional[float], low: float, high: float) -> str:
    """Return 'hard' if < low, 'medium' if in [low, high), 'easy' if >= high.
    Missing values default to 'hard'.
    """
    if pass_rate is None:
        return "hard"
    if pass_rate < low:
        return "hard"
    if pass_rate < high:
        return "medium"
    return "easy"


def process_file(
    input_path: Path,
    split: str,
    basis: str,
    thr7b: Tuple[float, float],
    thr30b: Tuple[float, float],
    buckets: Dict[str, List[Dict[str, Any]]],
) -> None:
    df = pd.read_parquet(input_path)
    domain = input_path.stem.split('__')[0]

    for idx, row in df.iterrows():
        if domain == 'math':
            rec = unify_math_data(row, idx, split)
        elif domain in ('codegen', 'code'):
            rec = unify_code_data(row, idx, split)
        elif domain == 'logic':
            rec = unify_logic_data(row, idx, split)
        elif domain == 'stem':
            rec = unify_stem_data(row, idx, split)
        elif domain == 'simulation':
            rec = unify_simulation_data(row, idx, split)
        else:
            continue

        pr_7b = _safe_get_float(row, "qwen2.5_7b_pass_rate")
        pr_30b = _safe_get_float(row, "qwen3_30b_pass_rate")

        if basis == '7b':
            low, high = thr7b
            chosen = pr_7b
        elif basis == '30b':
            low, high = thr30b
            chosen = pr_30b
        else:
            raise ValueError("basis must be '7b' or '30b'")

        difficulty = _classify_difficulty(chosen, low, high)
        buckets[difficulty].append(rec)


def main():
    parser = argparse.ArgumentParser(description="Convert to VeRL format and split by difficulty")
    parser.add_argument("--basis", choices=["7b", "30b"], default="7b",
                        help="Which pass-rate column to use for difficulty bucketing")

    parser.add_argument("--thr7b", default="0.3,0.6",
                        help="Thresholds for 7B as 'low,high' (hard < low <= medium < high <= easy)")
    parser.add_argument("--thr30b", default="0.4,0.7",
                        help="Thresholds for 30B as 'low,high' (hard < low <= medium < high <= easy)")
    parser.add_argument("--base_dir", default="/workspace/dev/Reasoning360/scripts/tools/data",
                        help="Base directory of source parquet files")
    parser.add_argument("--out_dir", default="../../data/guru_verl_level",
                        help="Output directory for VeRL parquets (three files)")
    args = parser.parse_args()

    thr7b = _parse_threshold_pair(args.thr7b)
    thr30b = _parse_threshold_pair(args.thr30b)

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    files = [
        ("train/math__combined_54.4k.parquet", "train"),
        ("train/logic__arcagi1_111.parquet", "train"),
        ("train/logic__arcagi2_190.parquet", "train"),
        ("train/logic__barc_1.6k.parquet", "train"),
        ("train/logic__graph_logical_1.2k.parquet", "train"),
        ("train/logic__ordering_puzzle_1.9k.parquet", "train"),
        ("train/logic__zebra_puzzle_1.3k.parquet", "train"),
        ("train/stem__web_3.6k.parquet", "train"),
        ("train/codegen__leetcode2k_1.3k.parquet", "train"),
        ("train/codegen__livecodebench_440.parquet", "train"),
        ("train/codegen__primeintellect_7.5k.parquet", "train"),
        ("train/codegen__taco_8.8k.parquet", "train"),
    ]

    buckets: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}

    for rel, split in files:
        inp = base_dir / rel
        if not inp.exists():
            continue
        process_file(
            input_path=inp,
            split=split,
            basis=args.basis,
            thr7b=thr7b,
            thr30b=thr30b,
            buckets=buckets,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("easy", "medium", "hard"):
        df_out = pd.DataFrame(buckets[name])
        out_path = out_dir / f"train_{name}.parquet"
        if len(df_out) == 0:
            # Create empty dataframe with minimal schema if needed
            df_out = pd.DataFrame([
                {"data_source": "", "prompt": [], "ability": "", "reward_model": {}, "extra_info": {}}
            ])
            df_out = df_out.iloc[0:0]
        df_out.to_parquet(out_path, engine="pyarrow")

if __name__ == '__main__':
    main()