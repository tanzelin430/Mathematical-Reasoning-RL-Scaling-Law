#!/usr/bin/env python3
"""
Extract cumulative compute (FLOPs) for each experiment from WandB logs
Enhanced to handle multiple model sizes and data sample sizes
Now also extracts critic/rewards/mean values for intrinsic performance calculation
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml

class ComputeExtractor:
    def __init__(self, wandb_root_dir="/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/RLscalewandb"):
        self.wandb_root_dir = Path(wandb_root_dir)
        self.model_size_map = {
            "0.5B": 0.5e9,
            "1.5B": 1.5e9,
            "3B": 3e9,
            "7B": 7e9,
        }
        
    def extract_step_tokens(self, log_file):
        """Extract (step, tokens) from log file"""
        step_tokens = []
        with open(log_file, 'r') as f:
            for line in f:
                step_match = re.search(r'step:(\d+)', line)
                token_match = re.search(r'perf/total_num_tokens:(\d+\.?\d*)', line)
                if step_match and token_match:
                    step = int(step_match.group(1))
                    tokens = float(token_match.group(1))
                    step_tokens.append((step, tokens))
        return step_tokens
    
    def extract_step_rewards(self, log_file):
        """Extract (step, critic/rewards/mean) from log file"""
        step_rewards = []
        with open(log_file, 'r') as f:
            for line in f:
                step_match = re.search(r'step:(\d+)', line)
                reward_match = re.search(r'critic/rewards/mean:([\d\.\-]+)', line)
                if step_match and reward_match:
                    step = int(step_match.group(1))
                    reward = float(reward_match.group(1))
                    step_rewards.append((step, reward))
        return step_rewards
    
    def get_run_timestamp(self, run_dir):
        """Extract timestamp from directory name"""
        match = re.search(r'offline-run-(\d{8}_\d{6})-', run_dir.name)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        return datetime.min
    
    def extract_config_info(self, run_dir):
        """Extract experiment info from config.yaml"""
        config_file = run_dir / "files" / "config.yaml"
        if not config_file.exists():
            return None, None
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract data sample size from training file path
            train_files = config.get('data', {}).get('value', {}).get('train_files', [])
            data_sample_size = None
            if train_files:
                train_file = train_files[0]
                print(f"    Parsing train_file: {train_file}")
                # Extract sample size from filename patterns:
                # ./data/guru_verl/difficulty_balanced_math/1000/math__difficulty_balanced_1000.parquet -> 1000
                # ./data/guru_verl/difficulty_balanced_math/100/math__difficulty_balanced_100_duplicated_to_512.parquet -> 100
                match = re.search(r'difficulty_balanced_math/(\d+)/', train_file)
                if not match:
                    # Fallback: try to extract from filename
                    match = re.search(r'difficulty_balanced_(\d+)', train_file)
                if match:
                    data_sample_size = int(match.group(1))
                    print(f"    Extracted data_sample_size: {data_sample_size}")
                else:
                    print(f"    No match found for: {train_file}")
            
            # Extract experiment name
            experiment_name = config.get('trainer', {}).get('value', {}).get('experiment_name', 'unknown')
            
            return data_sample_size, experiment_name
        except Exception as e:
            print(f"Error reading config from {config_file}: {e}")
            return None, None
    
    def extract_all_experiments(self):
        """Extract compute data for all experiments across all model sizes"""
        all_data = []
        
        # Process each model size directory
        for model_dir in self.wandb_root_dir.glob("wandb_*"):
            if not model_dir.is_dir():
                continue
                
            model_size = model_dir.name.replace("wandb_", "")
            if model_size not in self.model_size_map:
                print(f"Unknown model size: {model_size}")
                continue
                
            model_params = self.model_size_map[model_size]
            print(f"\n=== Processing {model_size} model ({model_params:.1e} params) ===")
            
            # Group by runid within this model size
            run_groups = {}
            for run_dir in model_dir.glob("offline-run-*"):
                if run_dir.is_dir():
                    runid = run_dir.name.split('-')[-1]
                    if runid not in run_groups:
                        run_groups[runid] = []
                    run_groups[runid].append(run_dir)
            
            # Sort each group by time
            for runid in run_groups:
                run_groups[runid].sort(key=self.get_run_timestamp)
            
            # Process each experiment within this model size
            for runid, run_dirs in run_groups.items():
                print(f"Processing {model_size} - {runid}: {len(run_dirs)} run(s)")
                
                # Get config info from the first run
                data_sample_size, experiment_name = self.extract_config_info(run_dirs[0])
                
                # Collect all steps from all runs (for resumed experiments)
                all_steps = []
                all_rewards = []
                for run_dir in run_dirs:
                    log_file = run_dir / "files" / "output.log"
                    if log_file.exists():
                        steps = self.extract_step_tokens(log_file)
                        rewards = self.extract_step_rewards(log_file)
                        all_steps.extend(steps)
                        all_rewards.extend(rewards)
                        print(f"  {run_dir.name}: {len(steps)} steps, {len(rewards)} reward points")
                
                if not all_steps:
                    continue
                    
                # Sort by step and calculate cumulative FLOPs
                all_steps.sort()
                cumulative_flops = 0
                
                # Create a dictionary for quick reward lookup
                reward_dict = dict(all_rewards)
                
                for step, tokens in all_steps:
                    # GRPO training FLOPs: 6x forward pass equivalent
                    step_flops = 6 * model_params * tokens
                    cumulative_flops += step_flops
                    
                    # Get reward for this step if available
                    reward = reward_dict.get(step, None)
                    
                    all_data.append({
                        'model_size': model_size,
                        'model_params': model_params,
                        'data_sample_size': data_sample_size,
                        'experiment_name': experiment_name,
                        'experiment_id': f"{model_size}_{runid}",
                        'runid': runid,
                        'step': step,
                        'tokens': tokens,
                        'step_flops': step_flops,
                        'cumulative_flops': cumulative_flops,
                        'critic_rewards_mean': reward
                    })
                
                print(f"  Total: {len(all_steps)} steps, sample_size: {data_sample_size}, final FLOPs: {cumulative_flops:.2e}")
        
        return pd.DataFrame(all_data)

def main():
    extractor = ComputeExtractor()
    print("=== Extracting Multi-Model Scaling Law Data (6x FLOPs multiplier) ===")
    
    df = extractor.extract_all_experiments()
    
    if df.empty:
        print("No data found!")
        return
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total experiments: {df['experiment_id'].nunique()}")
    print(f"Total data points: {len(df)}")
    
    # Group by model size
    print(f"\n=== By Model Size ===")
    for model_size in sorted(df['model_size'].unique(), key=lambda x: float(x.rstrip('B'))):
        model_data = df[df['model_size'] == model_size]
        unique_experiments = model_data['experiment_id'].nunique()
        unique_data_sizes = sorted(model_data['data_sample_size'].dropna().unique())
        print(f"  {model_size}: {unique_experiments} experiments, data sizes: {unique_data_sizes}")
    
    # Detailed experiment info
    print(f"\n=== Detailed Experiments ===")
    for exp_id in sorted(df['experiment_id'].unique()):
        exp_data = df[df['experiment_id'] == exp_id].iloc[0]  # Get first row for metadata
        max_step = df[df['experiment_id'] == exp_id]['step'].max()
        max_flops = df[df['experiment_id'] == exp_id]['cumulative_flops'].max()
        steps_count = len(df[df['experiment_id'] == exp_id])
        reward_count = df[df['experiment_id'] == exp_id]['critic_rewards_mean'].notna().sum()
        
        print(f"  {exp_id}: {steps_count} steps (max: {max_step}), "
              f"data_size: {exp_data['data_sample_size']}, FLOPs: {max_flops:.2e}, "
              f"rewards: {reward_count}")
    
    # Save
    output_file = '/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/scaling_law_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to {output_file}")
    
    return df

if __name__ == "__main__":
    df = main()
