train_files = [
        "math__combined_54.4k.parquet",
        "logic__arcagi1_111.parquet",
        "logic__arcagi2_190.parquet",
        "logic__barc_1.6k.parquet",
        "logic__graph_logical_1.2k.parquet",
        "logic__ordering_puzzle_1.9k.parquet",
        "logic__zebra_puzzle_1.3k.parquet",
        "stem__web_3.6k.parquet"
]
import pandas as pd
import json
import numpy as np

file = 'stem__web_3.6k.parquet'

# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

base_dir = '/workspace/dev/Reasoning360/scripts/tools/data/train'
df = pd.read_parquet(f'{base_dir}/{file}')
sample = df.head(1).to_dict('records')[0]

# Pretty print the sample
print(json.dumps(sample, indent=2, ensure_ascii=False, cls=NumpyEncoder))
print(f"\nTotal samples in this file: {len(df)}")
print(f"Columns: {list(df.columns)}")

base_dir = '/workspace/data/guru_verl/train'
df = pd.read_parquet(f'{base_dir}/{file}')
sample = df.head(1).to_dict('records')[0]

# Pretty print the sample
print(json.dumps(sample, indent=2, ensure_ascii=False, cls=NumpyEncoder))
print(f"\nTotal samples in this file: {len(df)}")
print(f"Columns: {list(df.columns)}")

