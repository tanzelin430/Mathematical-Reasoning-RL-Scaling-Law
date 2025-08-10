import os
import random
import numpy as np
import torch

def add_suffix(filename, sample_size):
    if sample_size < 1000:
        size_str = f"{sample_size}"
    elif (sample_size / 1000) % 1 != 0:
        size_str = f"{sample_size / 1000:.1f}k"
    else:
        size_str = f"{sample_size // 1000}k"
    return f"{filename}_{size_str}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def sample_dataset(dataset, sample_size):
    """
    Sample a dataset to a given size.
    """
    if sample_size is not None:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:min(sample_size, len(dataset))]
        dataset = dataset.select(indices)
    return dataset


def save_dataset(dataset, output_dir, filename_prefix, sample_size=None):
    """
    Save a dataset to a parquet file with appropriate naming.
    
    Args:
        dataset: The dataset to save
        output_dir: Directory to save the dataset
        filename_prefix: Base filename to use
        sample_size: Sample size to add as suffix to filename
    
    Returns:
        str: Path to the saved file
    """
    # Add suffix based on actual dataset size if sample_size is None
    if sample_size is None:
        sample_size = len(dataset)
    
    # Create filename with appropriate suffix
    filename = add_suffix(filename_prefix, sample_size)
    output_path = os.path.join(output_dir, f"{filename}.parquet")
    
    # Save dataset
    dataset.to_parquet(output_path)
    
    return output_path