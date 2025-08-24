#!/usr/bin/env python3
"""
Preprocess guru-RL-92k data directly into VeRL format.
Each row is converted to a dict with keys: data_source, prompt, ability, reward_model, extra_info.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List


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


def unify_code_data(row: pd.Series, idx: int, split: str, dataset_name: str) -> Dict[str, Any]:
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

    # Start from the row's original extra_info (may be dict or JSON string)
    extra_info = row.get('extra_info', {})
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
            extra_info = {}
    if not isinstance(extra_info, dict):
        extra_info = {}

    # Determine optional starter code prefix; only non-empty for LeetCode-like datasets (fuzzy match)
    prefix = ''
    if 'leetcode' in str(dataset_name).lower():
        existing_prefix = extra_info.get('prefix')
        if isinstance(existing_prefix, str):
            prefix = existing_prefix or ''

    # Based on Reasoning360: code domain doesn't use system prompt
    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": row.get("ability", "codegen"),  # Use 'codegen' to match the data
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {'split': split, 'index': idx, 'prefix': prefix}
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


def process_file(input_path: Path, output_path: Path, split: str) -> None:
    df = pd.read_parquet(input_path)
    unified: List[Dict[str, Any]] = []
    domain = input_path.stem.split('__')[0]

    for idx, row in df.iterrows():
        if domain == 'math':
            rec = unify_math_data(row, idx, split)
        elif domain in ('codegen', 'code'):
            rec = unify_code_data(row, idx, split, input_path.name)
        elif domain == 'logic':
            rec = unify_logic_data(row, idx, split)
        elif domain == 'stem':
            rec = unify_stem_data(row, idx, split)
        else:
            continue
        unified.append(rec)

    # Save as JSONL or parquet of dicts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unified).to_parquet(output_path, engine='pyarrow')


def main():
    base_dir = Path('/mnt/shared-storage-user/ma4agi-gpu/data/dataset/guru-RL-92k')
    files = [
        # Training data
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
        ("train/codegen__taco_8.8k.parquet", "train")
    ]

    for rel, split in files:
        inp = base_dir / rel
        out = Path('/home/tanzelin-p/Agentic-RL-Scaling-Law/data/guru_verl') / rel
        process_file(inp, out, split)

if __name__ == '__main__':
    main()