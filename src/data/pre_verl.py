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
    question = ''
    # Extract question content from prompt field
    if isinstance(row.get('prompt'), list) and row['prompt']:
        question = row['prompt'][0].get('content', '')
    elif pd.notna(row.get('prompt')):
        question = str(row['prompt'])

    solution = ''
    # Use direct ground_truth or solution field
    if pd.notna(row.get('ground_truth')):
        solution = str(row['ground_truth'])
    elif pd.notna(row.get('solution')):
        solution = str(row['solution'])

    data_source = row.get('data_source', 'math_unknown')

    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx}
    }
    return data


def unify_code_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    question = ''
    if pd.notna(row.get('prompt')):
        question = str(row['prompt'])
    elif 'query' in row and pd.notna(row['query']):
        question = str(row['query'])

    # Extract solution from test or ground_truth
    solution = ''
    if pd.notna(row.get('ground_truth')):
        solution = str(row['ground_truth'])
    elif pd.notna(row.get('test')):
        solution = str(row['test'])

    data_source = row.get('data_source', 'code_unknown')

    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx}
    }
    return data


def unify_logic_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    question = ''
    if isinstance(row.get('prompt'), list) and row['prompt']:
        question = row['prompt'][0].get('content', '')
    elif pd.notna(row.get('prompt')):
        question = str(row['prompt'])

    solution = ''
    if isinstance(row.get('reward_model'), dict):
        solution = str(row['reward_model'].get('ground_truth', ''))
    elif pd.notna(row.get('ground_truth')):
        solution = str(row['ground_truth'])

    data_source = row.get('data_source', 'logic_unknown')

    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "logic",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx}
    }
    return data


def unify_stem_data(row: pd.Series, idx: int, split: str) -> Dict[str, Any]:
    question = ''
    if pd.notna(row.get('prompt')):
        question = str(row['prompt'])
    elif pd.notna(row.get('raw_prompt')):
        question = str(row['raw_prompt'])

    solution = ''
    if isinstance(row.get('reward_model'), dict):
        solution = str(row['reward_model'].get('ground_truth', ''))

    data_source = row.get('data_source', 'stem_unknown')

    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "stem",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx}
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
            rec = unify_code_data(row, idx, split)
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
    base_dir = Path('/workspace/dev/Reasoning360/scripts/tools/data')
    files = [
        ("train/math__combined_54.4k.parquet", "train"),
        ("train/logic__arcagi1_111.parquet", "train"),
        ("train/logic__arcagi2_190.parquet", "train"),
        ("train/logic__barc_1.6k.parquet", "train"),
        ("train/logic__graph_logical_1.2k.parquet", "train"),
        ("train/logic__ordering_puzzle_1.9k.parquet", "train"),
        ("train/logic__zebra_puzzle_1.3k.parquet", "train"),
        ("train/stem__web_3.6k.parquet", "train"),
    ]

    for rel, split in files:
        inp = base_dir / rel
        out = Path('/workspace/data/guru_verl') / rel
        process_file(inp, out, split)

if __name__ == '__main__':
    main()