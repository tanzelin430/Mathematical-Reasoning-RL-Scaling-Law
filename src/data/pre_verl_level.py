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
    #yifanL change the math format
    # Based on Reasoning360: math domain doesn't use system prompt, just adds instruction
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
    
    return {
            "data_source": row.get("data_source", "math_unknown"),
            "prompt": prompt,
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
        # Get ground truth from reward_model
        solution = row['reward_model'].get('ground_truth', '')
        
        # Convert numpy arrays to lists for compatibility with scorers
        if isinstance(solution, np.ndarray):
            # Check if it's an array of arrays (ARC-AGI format)
            if len(solution) > 0 and isinstance(solution[0], np.ndarray):
                # Convert array of arrays to 2D list
                solution = [arr.tolist() for arr in solution]
            else:
                # Convert simple array to list
                solution = solution.tolist()
        elif not isinstance(solution, (list, dict, str)) and pd.notna(solution):
            # For other types, convert to string
            solution = str(solution)
    elif pd.notna(row.get('ground_truth')):
        solution = row['ground_truth']
        if isinstance(solution, np.ndarray):
            if len(solution) > 0 and isinstance(solution[0], np.ndarray):
                solution = [arr.tolist() for arr in solution]
            else:
                solution = solution.tolist()
        elif not isinstance(solution, (list, dict, str)):
            solution = str(solution)

    data_source = row.get('data_source', 'logic_unknown')

    # Based on Reasoning360: logic domain doesn't use system prompt
    data = {
        "data_source": data_source,
        "prompt": row.get("prompt", []),
        "ability": row.get("ability", "logic"),  # Keep the default as 'logic'
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx}
    }
    return data


def unify_stem_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    solution = ''
    if isinstance(row.get('reward_model'), dict):
        solution = str(row['reward_model'].get('ground_truth', ''))

    data_source = row.get('data_source', 'stem_unknown')

    prompt = row.get("prompt", [])
    
    # Based on Reasoning360: STEM domain adds instruction in user content
    STEM_INSTRUCTION = "You are a knowledgeable assistant. Answer the following questions and think step by step. Please output the final answer within \\boxed{}. "
    
    if prompt and len(prompt) > 0:
        prompt_list = prompt.tolist() if hasattr(prompt, 'tolist') else prompt
        if prompt_list and isinstance(prompt_list[0], dict) and prompt_list[0].get('role') == 'user':
            content = prompt_list[0].get('content', '')
            # Add STEM instruction at the beginning of user content
            if not content.startswith(STEM_INSTRUCTION):
                prompt_list[0]['content'] = STEM_INSTRUCTION + content
                prompt = np.array(prompt_list) if hasattr(prompt, 'tolist') else prompt_list
    
    # Extract question for stem_llm_judge
    question = ""
    if prompt and len(prompt) > 0:
        prompt_list = prompt.tolist() if hasattr(prompt, 'tolist') else prompt
        if prompt_list and isinstance(prompt_list[0], dict) and prompt_list[0].get('role') == 'user':
            content = prompt_list[0].get('content', '')
            # Remove the STEM instruction to get just the question
            if content.startswith(STEM_INSTRUCTION):
                question = content[len(STEM_INSTRUCTION):].strip()
            else:
                question = content.strip()
    
    data = {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "stem",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx, "question": question}
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
    out_dir: Path,
    mode: str = "threshold",
) -> None:
    df = pd.read_parquet(input_path)
    domain_raw = input_path.stem.split('__')[0]
    # Normalize domain name for consistency
    domain_key = 'codegen' if domain_raw in ('code', 'codegen') else domain_raw
    
    # Create buckets for this specific file
    file_buckets = {"easy": [], "medium": [], "hard": []}

    if mode == "equal":
        # Equal split mode: collect all records with their pass rates, then sort and split
        records_with_rates = []
        
        for idx, row in df.iterrows():
            if domain_key == 'math':
                rec = unify_math_data(row, idx, split)
            elif domain_key == 'codegen':
                rec = unify_code_data(row, idx, split)
            elif domain_key == 'logic':
                rec = unify_logic_data(row, idx, split)
            elif domain_key == 'stem':
                rec = unify_stem_data(row, idx, split)
            elif domain_key == 'simulation':
                rec = unify_simulation_data(row, idx, split)
            else:
                continue

            pr_7b = _safe_get_float(row, "qwen2.5_7b_pass_rate")
            pr_30b = _safe_get_float(row, "qwen3_30b_pass_rate")

            if basis == '7b':
                chosen_rate = pr_7b
            elif basis == '30b':
                chosen_rate = pr_30b
            else:
                raise ValueError("basis must be '7b' or '30b'")
            
            # Use -1 for None values to sort them at the end (hardest)
            sort_key = chosen_rate if chosen_rate is not None else -1
            records_with_rates.append((rec, sort_key))
        
        # Sort by pass rate (descending: high pass rate = easy)
        records_with_rates.sort(key=lambda x: x[1], reverse=True)
        
        # Split into three equal parts
        n = len(records_with_rates)
        third = n // 3
        
        file_buckets["easy"] = [rec for rec, _ in records_with_rates[:third]]
        file_buckets["medium"] = [rec for rec, _ in records_with_rates[third:2*third]]
        file_buckets["hard"] = [rec for rec, _ in records_with_rates[2*third:]]
        
    else:
        # Threshold mode: original logic
        for idx, row in df.iterrows():
            if domain_key == 'math':
                rec = unify_math_data(row, idx, split)
            elif domain_key == 'codegen':
                rec = unify_code_data(row, idx, split)
            elif domain_key == 'logic':
                rec = unify_logic_data(row, idx, split)
            elif domain_key == 'stem':
                rec = unify_stem_data(row, idx, split)
            elif domain_key == 'simulation':
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
            file_buckets[difficulty].append(rec)
    
    # Save this file's data into three difficulty folders
    original_filename = input_path.stem  # e.g., "logic__graph_logical_1.2k"
    
    for diff_name in ("easy", "medium", "hard"):
        diff_dir = out_dir / diff_name
        diff_dir.mkdir(parents=True, exist_ok=True)
        
        items = file_buckets[diff_name]
        if not items or len(items) == 0:
            print(f"⚠️  Skipping empty: {original_filename}_{diff_name} (0 examples)")
            continue
        
        try:
            df_out = pd.DataFrame(items)
            if len(df_out) == 0:
                print(f"⚠️  Skipping empty DataFrame: {original_filename}_{diff_name}")
                continue
                
            out_path = diff_dir / f"{original_filename}_{diff_name}.parquet"
            df_out.to_parquet(out_path, engine="pyarrow")
            print(f"💾 Saved {len(items)} examples to {out_path}")
        except Exception as e:
            print(f"❌ Error saving {original_filename}_{diff_name}: {e}")
            print(f"   Attempting to save as JSON instead...")
            try:
                out_path_json = diff_dir / f"{original_filename}_{diff_name}.json"
                df_out.to_json(out_path_json, orient='records', lines=True)
                print(f"💾 Saved {len(items)} examples to {out_path_json} (JSON format)")
            except Exception as e2:
                print(f"❌ Failed to save {original_filename}_{diff_name} in any format: {e2}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Convert to VeRL format and split by difficulty")
    parser.add_argument("--basis", choices=["7b", "30b"], default="7b",
                        help="Which pass-rate column to use for difficulty bucketing")
    parser.add_argument("--mode", choices=["threshold", "equal"], default="threshold",
                        help="Split mode: 'threshold' for pass-rate-based thresholds, 'equal' for equal-count split sorted by pass rate")

    parser.add_argument("--thr7b", default="0.3,0.6",
                        help="Thresholds for 7B as 'low,high' (hard < low <= medium < high <= easy)")
    parser.add_argument("--thr30b", default="0.4,0.7",
                        help="Thresholds for 30B as 'low,high' (hard < low <= medium < high <= easy)")
    parser.add_argument("--base_dir", default="/home/local/PARTNERS/yz646/Agentic-RL-Scaling-Law/dev/Reasoning360/scripts/tools/data",
                        help="Base directory of source parquet files")
    parser.add_argument("--out_dir", default="/home/local/PARTNERS/yz646/Agentic-RL-Scaling-Law/data/guru_verl_level",
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

    out_dir.mkdir(parents=True, exist_ok=True)
    
    for rel, split in files:
        inp = base_dir / rel
        if not inp.exists():
            print(f"⚠️  Skipping missing file: {inp}")
            continue
        print(f"📁 Processing: {rel}")
        process_file(
            input_path=inp,
            split=split,
            basis=args.basis,
            thr7b=thr7b,
            thr30b=thr30b,
            out_dir=out_dir,
            mode=args.mode,
        )
        print(f"✅ Completed: {rel}")
    
    print(f"\n🎉 Processing completed! Files saved in {out_dir}")
    print("📁 Output structure:")
    print("   easy/")
    print("   medium/") 
    print("   hard/")

if __name__ == '__main__':
    main()