import pandas as pd
import json
import numpy as np
from pathlib import Path

# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

base_dir = Path('/root/work/Agentic-RL-Scaling-Law/data/guru_verl/math')

# Get all parquet files in the directory
parquet_files = sorted(base_dir.glob('*.parquet'))

print(f"Found {len(parquet_files)} parquet files in {base_dir}\n")
print("=" * 80)

# Collect all data sources
all_data_sources = {}

# Process each file
for file_path in parquet_files:
    file_name = file_path.name
    print(f"\n📁 FILE: {file_name}")
    print("-" * 60)
    
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        # Get the first sample
        if len(df) > 0:
            sample = df.head(1).to_dict('records')[0]
            
            # Pretty print the sample
            print("Sample data:")
            print(json.dumps(sample, indent=2, ensure_ascii=False, cls=NumpyEncoder))
            
            # Print statistics
            print(f"\n📊 Statistics:")
            print(f"  - Total samples: {len(df)}")
            print(f"  - Columns: {list(df.columns)}")
            
            # Collect data sources
            if 'data_source' in df.columns:
                for source in df['data_source'].unique():
                    if source not in all_data_sources:
                        all_data_sources[source] = []
                    all_data_sources[source].append(file_name)
            
            # Show domain distribution if 'ability' column exists
            if 'ability' in df.columns:
                domain_counts = df['ability'].value_counts()
                print(f"  - Domain distribution:")
                for domain, count in domain_counts.items():
                    print(f"    • {domain}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print("⚠️  File is empty!")
            
    except Exception as e:
        print(f"❌ Error processing file: {e}")
    
    print("=" * 80)

# Print summary of all data sources
print("\n" + "=" * 80)
print("📚 SUMMARY: ALL DATA SOURCES")
print("=" * 80)

if all_data_sources:
    for source in sorted(all_data_sources.keys()):
        print(f"\n📌 {source}:")
        for file_name in all_data_sources[source]:
            print(f"    → {file_name}")
else:
    print("No data sources found in the files.")

print("\n" + "=" * 80)

