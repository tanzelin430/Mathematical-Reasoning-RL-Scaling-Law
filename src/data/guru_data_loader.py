"""
Data loader for 4 domains from guru-RL-92k dataset
Domains: math, code, science, logic
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GuruFourDomainDataLoader:
    """Load and mix data from 4 specific domains in guru-RL-92k"""
    
    SELECTED_DOMAINS = ['math', 'codegen', 'science', 'logic']
    
    def __init__(self, data_dir: str = "/fs-computility/mabasic/shared/data/guru-RL-92k"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.eval_dir = self.data_dir / "online_eval"
        
    def get_domain_files(self, split: str = "train") -> Dict[str, List[str]]:
        """Get parquet files for each domain"""
        base_dir = self.train_dir if split == "train" else self.eval_dir
        
        domain_files = {}
        for domain in self.SELECTED_DOMAINS:
            # Handle both naming conventions
            pattern1 = base_dir / f"{domain}__*.parquet"
            pattern2 = base_dir / f"{domain}_*.parquet"
            
            files = list(glob.glob(str(pattern1))) + list(glob.glob(str(pattern2)))
            
            if files:
                domain_files[domain] = sorted(files)
                logger.info(f"Found {len(files)} files for {domain} in {split}")
            else:
                logger.warning(f"No files found for {domain} in {split}")
                
        return domain_files
    
    def load_domain_data(self, domain: str, files: List[str], max_samples: Optional[int] = None) -> pd.DataFrame:
        """Load data from a specific domain"""
        dfs = []
        total_samples = 0
        
        for file in files:
            df = pd.read_parquet(file)
            
            # Add domain label if not present
            if 'domain' not in df.columns:
                df['domain'] = domain
                
            dfs.append(df)
            total_samples += len(df)
            
            if max_samples and total_samples >= max_samples:
                break
                
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if max_samples:
            combined_df = combined_df.head(max_samples)
            
        return combined_df
    
    def load_mixed_data(self, split: str = "train", 
                       max_samples_per_domain: Optional[Dict[str, int]] = None,
                       seed: int = 42) -> pd.DataFrame:
        """Load and mix data from all 4 domains"""
        domain_files = self.get_domain_files(split)
        
        all_data = []
        domain_stats = {}
        
        for domain, files in domain_files.items():
            if not files:
                continue
                
            max_samples = None
            if max_samples_per_domain and domain in max_samples_per_domain:
                max_samples = max_samples_per_domain[domain]
                
            domain_df = self.load_domain_data(domain, files, max_samples)
            all_data.append(domain_df)
            
            domain_stats[domain] = len(domain_df)
            
        # Combine all domains
        mixed_df = pd.concat(all_data, ignore_index=True)
        
        # Shuffle with fixed seed for reproducibility
        mixed_df = mixed_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Log statistics
        logger.info(f"Loaded {len(mixed_df)} total samples from {split}")
        for domain, count in domain_stats.items():
            percentage = (count / len(mixed_df)) * 100
            logger.info(f"  {domain}: {count} samples ({percentage:.1f}%)")
            
        return mixed_df
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the loaded data"""
        stats = {
            'total_samples': len(df),
            'domains': {}
        }
        
        # Domain distribution
        if 'domain' in df.columns:
            domain_counts = df['domain'].value_counts()
            for domain, count in domain_counts.items():
                stats['domains'][domain] = {
                    'count': int(count),
                    'percentage': float(count / len(df) * 100)
                }
                
        # Average lengths
        if 'prompt' in df.columns:
            stats['avg_prompt_length'] = df['prompt'].str.len().mean()
            
        if 'data_source' in df.columns:
            # More detailed source breakdown
            source_counts = df['data_source'].value_counts()
            stats['detailed_sources'] = source_counts.to_dict()
            
        return stats
    
    def prepare_for_training(self, df: pd.DataFrame) -> List[Dict]:
        """Convert dataframe to training format"""
        training_data = []
        
        for _, row in df.iterrows():
            sample = {
                'prompt': row['prompt'],
                'domain': row.get('domain', 'unknown'),
                'data_source': row.get('data_source', ''),
            }
            
            # Include ground truth if available
            if 'ground_truth' in row:
                sample['ground_truth'] = row['ground_truth']
                
            # Include other metadata
            for col in ['difficulty', 'category', 'subcategory']:
                if col in row:
                    sample[col] = row[col]
                    
            training_data.append(sample)
            
        return training_data


def get_train_val_files(data_loader: GuruFourDomainDataLoader) -> tuple:
    """Get train and validation file lists for verl config"""
    train_files = []
    val_files = []
    
    # Get training files
    train_domain_files = data_loader.get_domain_files("train")
    for domain, files in train_domain_files.items():
        train_files.extend(files)
        
    # Get validation files  
    val_domain_files = data_loader.get_domain_files("online_eval")
    for domain, files in val_domain_files.items():
        val_files.extend(files)
        
    return train_files, val_files


if __name__ == "__main__":
    # Test the data loader
    loader = GuruFourDomainDataLoader()
    
    # Load training data
    print("Loading training data...")
    train_df = loader.load_mixed_data("train")
    train_stats = loader.get_data_statistics(train_df)
    
    print("\nTraining data statistics:")
    print(f"Total samples: {train_stats['total_samples']}")
    for domain, info in train_stats['domains'].items():
        print(f"  {domain}: {info['count']} ({info['percentage']:.1f}%)")
        
    # Load validation data
    print("\nLoading validation data...")
    val_df = loader.load_mixed_data("online_eval")
    val_stats = loader.get_data_statistics(val_df)
    
    print("\nValidation data statistics:")
    print(f"Total samples: {val_stats['total_samples']}")
    for domain, info in val_stats['domains'].items():
        print(f"  {domain}: {info['count']} ({info['percentage']:.1f}%)")