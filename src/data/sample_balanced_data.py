#!/usr/bin/env python3
"""
Sample balanced data from each domain for training.
Each domain will have the same number of samples.
Keeps files separate to avoid format mixing issues.
"""
import argparse
import pandas as pd
from pathlib import Path
import random
import sys

def main():
    parser = argparse.ArgumentParser(description='Sample balanced data from each domain')
    parser.add_argument('--total_samples', type=int, required=True, help='Total number of samples')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for sampled data')
    args = parser.parse_args()
    
    # Calculate samples per domain
    samples_per_domain = args.total_samples // 4
    print(f"Total samples: {args.total_samples}")
    print(f"Samples per domain: {samples_per_domain}")
    
    # Define data files for each domain
    base_dir = Path("/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/data/guru_verl/train")
    
    domain_files = {
        "math": ["math__combined_54.4k.parquet"],
        "code": [
            "codegen__leetcode2k_1.3k.parquet",
            "codegen__livecodebench_440.parquet", 
            "codegen__primeintellect_7.5k.parquet",
            "codegen__taco_8.8k.parquet"
        ],
        "logic": [
            "logic__arcagi1_111.parquet",
            "logic__arcagi2_190.parquet",
            "logic__barc_1.6k.parquet",
            "logic__graph_logical_1.2k.parquet",
            "logic__ordering_puzzle_1.9k.parquet",
            "logic__zebra_puzzle_1.3k.parquet"
        ],
        "stem": ["stem__web_3.6k.parquet"]
    }
    
    # Create output directory structure
    output_path = Path(args.output_dir)
    
    # Track all output files
    all_output_files = []
    
    for domain, files in domain_files.items():
        print(f"\nProcessing {domain} domain...")
        
        # Create domain subdirectory
        domain_dir = output_path / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate total samples available for this domain
        total_available = 0
        file_info = []
        for file in files:
            file_path = base_dir / file
            df_size = len(pd.read_parquet(file_path))
            total_available += df_size
            file_info.append((file, file_path, df_size))
        
        print(f"  Total available samples: {total_available}")
        
        # Check if we have enough samples
        if total_available < samples_per_domain:
            print(f"\nERROR: {domain} domain has only {total_available} samples, but {samples_per_domain} are required!")
            print(f"Please reduce total_samples or add more data to {domain} domain.")
            sys.exit(1)
        
        # Distribute samples proportionally across files
        remaining_samples = samples_per_domain
        for i, (file, file_path, df_size) in enumerate(file_info):
            # For the last file, take all remaining samples
            if i == len(file_info) - 1:
                samples_from_this_file = remaining_samples
            else:
                # Calculate proportional samples
                proportion = df_size / total_available
                samples_from_this_file = min(int(samples_per_domain * proportion * 1.2), df_size, remaining_samples)
            
            if samples_from_this_file > 0:
                # Read and sample from this file
                df = pd.read_parquet(file_path)
                sampled = df.sample(n=min(samples_from_this_file, len(df)), random_state=42)
                
                # Save with original filename pattern
                output_filename = file.replace('.parquet', f'_sampled_{len(sampled)}.parquet')
                output_file = domain_dir / output_filename
                sampled.to_parquet(output_file, engine='pyarrow')
                all_output_files.append(output_file)
                
                print(f"    Sampled {len(sampled)} from {file} -> {output_filename}")
                
                remaining_samples -= len(sampled)
                if remaining_samples <= 0:
                    break
    
    print(f"\nSuccessfully created balanced dataset with {args.total_samples} total samples")
    print(f"Output files saved to: {output_path}")
    
    # Print file list for easy copying to training script
    print("\nTraining file list:")
    print("[")
    for f in all_output_files:
        print(f"    '{f}',")
    print("]")

if __name__ == '__main__':
    main()