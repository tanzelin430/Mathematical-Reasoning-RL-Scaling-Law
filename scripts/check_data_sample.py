import pandas as pd
import json
import numpy as np

# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

# Load a sample from the dataset
df = pd.read_parquet('/fs-computility/mabasic/shared/data/guru-RL-92k/train/codegen__livecodebench_440.parquet')
sample = df.head(1).to_dict('records')[0]

# Pretty print the sample
print(json.dumps(sample, indent=2, ensure_ascii=False, cls=NumpyEncoder))
print(f"\nTotal samples in this file: {len(df)}")
print(f"Columns: {list(df.columns)}")