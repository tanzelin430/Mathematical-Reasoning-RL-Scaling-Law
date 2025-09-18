#!/usr/bin/env python3
"""
Extract cumulative compute (FLOPs) for each experiment from Experiment2 logs
Enhanced to handle multiple model sizes and extract test scores instead of training rewards
Now extracts val/test_score/math__math/unknown values for intrinsic performance calculation
Handles new directory structure with run_X subdirectories and extracts slice_factor
"""

import re
import pandas as pd
from pathlib import Path
import data_proc

class ComputeExtractorExperiment:
    def __init__(self):
        self.model_size_map = {
            "0.5B": 0.5e9,
            "1.5B": 1.5e9,
            "3B": 3e9,
            "7B": 7e9,
            "14B": 14e9,
            "0.5b": 0.5e9,
            "1.5b": 1.5e9,
            "3b": 3e9,
            "7b": 7e9,
            "14b": 14e9,
        }
        
    def extract_step_tokens(self, log_file):
        """Extract (step, tokens) from log file"""
        step_tokens = []
        with open(log_file, 'r') as f:
            for line in f:
                step_match = re.search(r'step:(\d+)', line)
                token_match = re.search(r'perf/total_num_tokens:(\d+\.?\d*)', line)
                if step_match:
                    step = int(step_match.group(1))
                    if token_match:
                        tokens = float(token_match.group(1))
                        step_tokens.append((step, tokens))
                    if step == 0:
                        tokens = 0
                        step_tokens.append((step, tokens))
        return step_tokens
    
    def extract_step_test_scores(self, log_file):
        """Extract all test scores from log file"""
        step_scores = []
        
        # Define all test score patterns
        test_score_patterns = [
            'val/test_score/math__deepscaler_preview/unknown',
            'val/test_score/math__merged_deduped_dapo_or1_dataset/unknown', 
            'val/test_score/math__math/unknown',
            'val/test_score/logic__zebra_puzzle_dataset/unknown',
            'val/test_score/stem__supergpqa/unknown',
            'val/test_score/codegen__humaneval/unknown',
            'val/test_score/aime2024/unknown',
            'val/test_score/openai/gsm8k/unknown',
            'val/test_score/aimeamc2023/unknown'
        ]
        
        # Also include overall_pass1
        overall_pattern = 'val/overall_pass1'
        
        with open(log_file, 'r') as f:
            for line in f:
                step_match = re.search(r'step:(\d+)', line)
                if not step_match:
                    continue
                    
                step = int(step_match.group(1))
                scores_dict = {}
                
                # Extract all test scores
                for pattern in test_score_patterns:
                    # Extract the eval name (everything after val/test_score/ and before /unknown)
                    eval_name = pattern.replace('/unknown', '')#.replace('val/test_score/', '')
                    score_match = re.search(f'{re.escape(pattern)}:([\d\.\-]+)', line)
                    if score_match:
                        scores_dict[eval_name] = float(score_match.group(1))
                
                # Extract overall_pass1
                overall_match = re.search(f'{re.escape(overall_pattern)}:([\d\.\-]+)', line)
                if overall_match:
                    scores_dict['overall_pass1'] = float(overall_match.group(1))
                
                # Only add if we found at least one score
                if scores_dict:
                    scores_dict['step'] = step
                    step_scores.append(scores_dict)
        
        return step_scores
    
    def extract_step_flops(self, log_file):
        """Extract (step, step_flops) from log file"""
        step_flops = []
        with open(log_file, 'r') as f:
            for line in f:
                step_match = re.search(r'step:(\d+)', line)
                flops_match = re.search(r'perf/total_num_tokens:(\d+\.?\d*)', line)
                if step_match and flops_match:
                    step = int(step_match.group(1))
                    tokens = float(flops_match.group(1))
                    # Calculate FLOPs: 6 * N * tokens (assuming 6 FLOPs per token)
                    # We'll need to get N from the model size, so we'll calculate this later
                    step_flops.append((step, tokens))
        return step_flops
    
    def get_model_size_from_path(self, log_path):
        """Extract model size from the path structure"""
        path_parts = Path(log_path).parts
        for part in path_parts:
            if part in self.model_size_map:
                return part
        return None
    
    def get_run_id_from_path(self, log_path):
        """Extract run ID from the path structure"""
        path_parts = Path(log_path).parts
        for part in path_parts:
            if part.startswith('offline-run-'):
                parts = part.split('-')
                if len(parts) > 1:
                    return parts[-1]
                return part
        return None
    
    def get_slice_factor_from_path(self, log_path):
        """Extract slice factor from the path structure (run_X directory)"""
        path_parts = Path(log_path).parts
        for part in path_parts:
            if part.startswith('run_'):
                try:
                    # Extract the number after 'run_'
                    slice_factor = int(part.split('_')[1])
                    return slice_factor
                except (ValueError, IndexError):
                    continue
        # If no run_X directory found, return default slice factor of 0
        return 0
    
    def process_single_experiment(self, log_file):
        """Process a single experiment log file"""
        # print(f"Processing: {log_file}")
        
        # Extract model size, run ID, and slice factor from path
        model_size = self.get_model_size_from_path(log_file)
        run_id = self.get_run_id_from_path(log_file)
        slice_factor = self.get_slice_factor_from_path(log_file)
        
        if not model_size or not run_id:
            print(f"  ⚠️  Could not extract model_size or run_id from path: {log_file}")
            return None
        
        # slice_factor will always be returned (default 1 if no run_X directory)
        
        model_params = self.model_size_map[model_size]
        
        # Extract data
        step_tokens = self.extract_step_tokens(log_file)
        step_scores = self.extract_step_test_scores(log_file)
        
        if not step_tokens:
            print(f"  ⚠️  No token data found in {log_file}")
            return None
        
        if not step_scores:
            print(f"  ⚠️  No test score data found in {log_file}")
            return None
        
        # print(f"  Found {len(step_tokens)} token records, {len(step_scores)} test score records")
        
        # Create DataFrames
        tokens_df = pd.DataFrame(step_tokens, columns=['step', 'tokens'])
        scores_df = pd.DataFrame(step_scores)
        
        # Merge on step, keeping only steps that have both tokens and test scores
        merged_df = pd.merge(tokens_df, scores_df, on='step', how='inner')
        
        # Calculate holdout_score as weighted average of two math datasets
        # Weight: deduped_dapo * 335 + deepscaler_preview * 165
        deduped_col = 'val/test_score/math__merged_deduped_dapo_or1_dataset'
        deepscaler_col = 'val/test_score/math__deepscaler_preview'
        
        if deduped_col in merged_df.columns and deepscaler_col in merged_df.columns:
            # Calculate weighted average: (deduped * 335 + deepscaler * 165) / (335 + 165)
            merged_df['holdout_score'] = (
                merged_df[deduped_col] * 335 + 
                merged_df[deepscaler_col] * 165
            ) / (335 + 165)
            # print(f"  ✅ Added holdout_score (weighted average of math datasets)")
        else:
            print(f"  ⚠️  Missing required columns for holdout_score calculation")
            if deduped_col not in merged_df.columns:
                print(f"    Missing: {deduped_col}")
            if deepscaler_col not in merged_df.columns:
                print(f"    Missing: {deepscaler_col}")
        
        if merged_df.empty:
            print(f"  ⚠️  No matching steps between tokens and test scores in {log_file}")
            return None
        
        # print(f"  Merged to {len(merged_df)} steps with both tokens and test scores")
        # print(f"  Available evals: {[col for col in merged_df.columns if col not in ['step', 'tokens']]}")
        
        # Calculate FLOPs: 6 * N * tokens
        merged_df['step_flops'] = 6 * model_params * merged_df['tokens']
        merged_df['cumulative_flops'] = merged_df['step_flops'].cumsum()
        
        # Calculate cumulative tokens
        merged_df['cumulative_tokens'] = merged_df['tokens'].cumsum()
        
        # Add metadata
        merged_df['model_size'] = model_size
        merged_df['model_params'] = model_params
        merged_df['data_sample_size'] = None  # Not available in this dataset
        merged_df['experiment_name'] = f"Qwen2.5-{model_size}_math_test_grpo_verl"
        merged_df['experiment_id'] = f"{model_size}_{run_id}"
        merged_df['runid'] = run_id
        merged_df['slice_factor'] = slice_factor
        
        # For backward compatibility, use math__math as the main R value (critic_rewards_mean)
        if 'math__math' in merged_df.columns:
            merged_df['critic_rewards_mean'] = merged_df['math__math']
        else:
            # Fallback to overall_pass1 if math__math is not available
            merged_df['critic_rewards_mean'] = merged_df.get('overall_pass1', 0.0)
        
        # Reorder columns: metadata first, then test evals, then compute data
        metadata_cols = ['model_size', 'model_params', 'data_sample_size', 'experiment_name', 
                        'experiment_id', 'runid', 'slice_factor', 'step']
        compute_cols = ['tokens', 'cumulative_tokens', 'step_flops', 'cumulative_flops', 'critic_rewards_mean', 'holdout_score']
        
        # Get all test eval columns (exclude step and tokens)
        test_eval_cols = [col for col in merged_df.columns 
                           if col not in metadata_cols + compute_cols + ['step', 'tokens']]
        
        # Reorder columns
        result_df = merged_df[metadata_cols + test_eval_cols + compute_cols].copy()
        
        return result_df
    
    def find_all_log_files(self):
        """Find all output.log files in the experiment directory"""
        log_files = []
        for model_size_dir in self.experiment_root_dir.iterdir():
            if not model_size_dir.is_dir():
                continue
            
            model_size = model_size_dir.name
            if model_size not in self.model_size_map:
                print(f"⚠️  Unknown model size directory: {model_size}")
                continue
            
            # Handle different directory structures
            for run_dir in model_size_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                # Check if this is a run_X directory
                if run_dir.name.startswith('run_'):
                    # Handle structure: model_size/run_X/offline-run-.../files/output.log
                    for offline_run_dir in run_dir.iterdir():
                        if not offline_run_dir.is_dir():
                            continue
                        
                        if offline_run_dir.name.startswith('offline-run-'):
                            log_file = offline_run_dir / "files" / "output.log"
                            if log_file.exists():
                                log_files.append(log_file)
                            else:
                                print(f"⚠️  Log file not found: {log_file}")
                elif run_dir.name.startswith('offline-run-'):
                    # Handle structure: model_size/offline-run-.../files/output.log
                    log_file = run_dir / "files" / "output.log"
                    if log_file.exists():
                        log_files.append(log_file)
                    else:
                        print(f"⚠️  Log file not found: {log_file}")
                else:
                    print(f"⚠️  Unknown directory structure: {run_dir}")
        
        return log_files
    
    def run(self, experiment_root_dir):
        """Main processing function"""
        print(f"\n=== Starting \"{experiment_root_dir}\" ===")

        # Set experiment root directory
        script_dir = Path(__file__).resolve().parent
        input_path = Path(experiment_root_dir).expanduser()
        if not input_path.is_absolute():
            input_path = (script_dir / input_path).resolve()
        self.experiment_root_dir = input_path
        print(f"root directory: {self.experiment_root_dir}")
        
        # Find all log files
        log_files = self.find_all_log_files()
        print(f"Found {len(log_files)} log files")
        
        if not log_files:
            raise ValueError("❌ No log files found!")
        
        # Process each experiment
        all_data = []
        for log_file in log_files:
            result_df = self.process_single_experiment(log_file)
            if result_df is not None:
                all_data.append(result_df)
        
        if not all_data:
            raise ValueError("❌ No valid data found!")
        
        # Combine all data
        df = pd.concat(all_data, ignore_index=True)
        
        # Sort by model size and step
        df = df.sort_values(['model_size', 'runid', 'step']).reset_index(drop=True)
        
        # Summary statistics are now handled in inspect() function
        
        
        self.df = df
        return self

    def inspect(self):
        self.df = self.df.sort_values(['model_size','runid','step']).reset_index(drop=True)
        data_proc.inspect_data(self.df)
        return self

    def save(self, output_file):
        # Save
        script_dir = Path(__file__).resolve().parent
        output_file = (script_dir / output_file).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_file, index=False)
        print(f"\n✅ Saved to {output_file}")
        return self

if __name__ == "__main__":
    # (ComputeExtractorExperiment()
    #  .run(experiment_root_dir='data/Experiment1_instruct/Experiment1_instruct_run0')
    #  .inspect()
    #  .save('csv/scaling_law_data_experiment1_instruct_run0.csv'))
    
    # (ComputeExtractorExperiment()
    #  .run(experiment_root_dir='data/Experiment1_instruct/Experiment1_instruct_run1')
    #  .inspect()
    #  .save('csv/scaling_law_data_experiment1_instruct_run1.csv'))

    # (ComputeExtractorExperiment()
    #  .run(experiment_root_dir='data/Experiment1_instruct/Experiment1_instruct_run2')
    #  .inspect()
    #  .save('csv/scaling_law_data_experiment1_instruct_run2.csv'))
    
    # (ComputeExtractorExperiment()
    #  .run(experiment_root_dir='data/Experiment1_Base_run0')
    #  .inspect()
    #  .save('csv/scaling_law_data_experiment1_base_run0.csv'))


    (ComputeExtractorExperiment()
     .run(experiment_root_dir='data/experiment2-base')
     .inspect()
     .save('csv/scaling_law_data_experiment2_base.csv'))

    # # Process experiment2-base data
    # (ComputeExtractorExperiment()
    #  .run(experiment_root_dir='data/experiment2-base')
    #  .inspect()
    #  .save('csv/scaling_law_data_experiment2_base.csv'))
