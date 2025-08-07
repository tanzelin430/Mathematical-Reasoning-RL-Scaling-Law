train_files = [
        "math__combined_54.4k.parquet",
        "logic__arcagi1_111.parquet",
        "logic__arcagi2_190.parquet",
        "logic__barc_1.6k.parquet",
        "logic__graph_logical_1.2k.parquet",
        "logic__ordering_puzzle_1.9k.parquet",
        "logic__zebra_puzzle_1.3k.parquet",
        "stem__web_3.6k.parquet",
        "codegen__leetcode2k_1.3k.parquet",
        "codegen__livecodebench_440.parquet",
        "codegen__primeintellect_7.5k.parquet",
        "codegen__taco_8.8k.parquet"
]
import pandas as pd
import json
import numpy as np

file = 'codegen__primeintellect_7.5k.parquet'

# Custom JSON encoder to handle numpy arrays and pandas types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if pd.isna(obj):
            return None
        if hasattr(obj, 'isoformat'):  # Handle datetime/timestamp objects
            return obj.isoformat()
        if hasattr(obj, 'item'):  # Handle other numpy scalars
            return obj.item()
        try:
            return str(obj)  # Fallback to string representation
        except:
            return f"<{type(obj).__name__} object>"

base_dir = '/workspace/dev/Reasoning360/scripts/tools/data/train'
df = pd.read_parquet(f'{base_dir}/{file}')
sample = df.head(1).to_dict('records')[0]

# Pretty print the sample
print(json.dumps(sample, indent=2, ensure_ascii=False, cls=NumpyEncoder))
print(f"\nTotal samples in this file: {len(df)}")
print(f"Columns: {list(df.columns)}")

# base_dir = '/workspace/data/guru_verl/train'
# df = pd.read_parquet(f'{base_dir}/{file}')
# sample = df.head(1).to_dict('records')[0]

# # Pretty print the sample
# print(json.dumps(sample, indent=2, ensure_ascii=False, cls=NumpyEncoder))
# print(f"\nTotal samples in this file: {len(df)}")
# print(f"Columns: {list(df.columns)}")

