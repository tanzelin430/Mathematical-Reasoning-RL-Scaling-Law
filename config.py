from pathlib import Path
import os
import sys
# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION - Edit these variables to customize the run
# =============================================================================

# Data source configuration - use absolute paths based on script location
SCRIPT_DIR = Path(__file__).parent
OUTPUT_BASE_DIR = SCRIPT_DIR / "outputs"  # Base output directory for PNG plots
SAMPLE_SIZE_PER_STEP = 512
BUILD_I_ON_SMOOTHED = True
# warmup clipping: Important even for LLM RL
WARMUP_CLIPPING_FACTOR_FOR_RAW = 10/100  # on raw data
WARMUP_CLIPPING_FACTOR_FOR_SMOOTH = 0/100  # only affects smoothed lines, not fitting lines, [based on clipped raw]

# HOLDOUT=True,
# Test evals to process (from the CSV columns)
DEFAULT_TEST_EVAL = 'holdout_score'
DEFAULT_FIGURE_PREFIX = 'holdout'
# DEFAULT_FIGURE_COLUMNS = 1 # note: if total > figure_columns, [row, col] -> [i]
# DEFAULT_FIGURE_SIZE=(5, 5)

# Test evals to process (from the CSV columns)
TEST_EVALS = {
    'holdout_score': {'file_str': 'holdout', 'plot_str': 'Holdout Validation'},
    'overall_pass1': {'file_str': 'overall_pass1', 'plot_str': 'Overall@Pass1'},
    'val/test_score/openai/gsm8k': {'file_str': 'gsm8k', 'plot_str': 'GSM8K'},
    'val/test_score/codegen__humaneval': {'file_str': 'codegen__humaneval', 'plot_str': 'CodeGen - HumanEval'},
    'val/test_score/stem__supergpqa': {'file_str': 'stem__supergpqa', 'plot_str': 'SuperGPQA'},
    'val/test_score/math__math': {'file_str': 'math__math', 'plot_str': 'Math'},
    'val/test_score/logic__zebra_puzzle_dataset': {'file_str': 'logic__zebra_puzzle_dataset', 'plot_str': 'Logic - Zebra Puzzle'},
    'val/test_score/aimeamc2023': {'file_str': 'aimeamc2023', 'plot_str': 'AMC 2023'},
    'val/test_score/aime2024': {'file_str': 'aime2024', 'plot_str': 'AMC 2024'},
    # 'val/test_score/math__deepscaler_preview': {'file_str': 'math__deepscaler_preview', 'plot_str': 'DeepScaler Preview'},
    # 'val/test_score/math__merged_deduped_dapo_or1_dataset': {'file_str': 'merged_deduped_dapo', 'plot_str': 'Merged Deduped Dapo OR1 Dataset'},
}
MULTI_FIGURE_COLUMNS = 2 # note: if total > figure_columns, [row, col] -> [i]
MULTI_FIGURE_SIZE=(10, 10)

TOTAL_EVALS = len(TEST_EVALS.keys())


DEFAULT_LABELS = {
    # Metrics (通常用作Y轴)
    'R': "Reward", 
    'ErrRate': "Test Loss", 
    'DeltaReward': "Improvement",
    'DeltaErrRate': "Delta Test Loss",
    # Variables (通常用作X轴或分组)
    "T": "Tokens",
    "C": "Compute (FLOPs)",
    "C_raw": "Compute (FLOPs)",
    "E": "Data Size",
    "N": "Model Size",
    "model_params": "Model Size",
    "DR": "Data Repeat",
    "slice_factor": "Data Repeat"
}

DEFAULT_SHORT_NAME = {
    "C_raw": "C",
    "E": "D",
    "slice_factor": "DR"
}

# 列名映射 - 标准化数据列名
COLUMN_RENAME_MAP = {
    'model_params': 'N',
    'cumulative_flops': 'C_raw',
    'runid': 'runid',
    'step': 'step',
    'tokens': 'tokens',
    'cumulative_tokens': 'T',
    'slice_factor': 'DR',
}

DEBUG = False

CSV_INSTRUCT_RUNS = [
        SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run0.csv" ,
        SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run1.csv" ,
        SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run2.csv" ,
    ]

CSV_BASE_RUNS = [
    SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_base.csv" ,
    SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_base_run0.csv" ,
]

CSV_LLAMA_BASE_RUNS = [
    SCRIPT_DIR / "csv" / "scaling_law_data_experiment-llama-base.csv" ,
]

CSV_LLAMA_INSTRUCT_RUNS = [
    SCRIPT_DIR / "csv" / "scaling_law_data_experiment-llama-instruct.csv" ,
]


CSV_EXPERIMENT2_BASE_RUNS = [
    SCRIPT_DIR / "csv" / "scaling_law_data_experiment2_base.csv" ,
]

CSV_EXPERIMENT2_INSTRUCT_RUNS = [
    SCRIPT_DIR / "csv" / "scaling_law_data_experiment2_instruct.csv" ,
]

CSV_MAP = {
    "base": CSV_BASE_RUNS,
    "instruct": CSV_INSTRUCT_RUNS,
    "llama-base": CSV_LLAMA_BASE_RUNS,
    "llama-instruct": CSV_LLAMA_INSTRUCT_RUNS,
    "exp2-base": CSV_EXPERIMENT2_BASE_RUNS,
    "exp2-instruct": CSV_EXPERIMENT2_INSTRUCT_RUNS,
}

# =============================================================================
# COLOR MAPPING - 统一的渐变配色方案
# =============================================================================

COLOR_MAPPING = {
    # for model size (从小到大：浅到深)
    0.5e9: '#aadc32',  # 黄绿色 (最小)
    1e9: '#aadc32',    # 黄绿色
    1.5e9: '#5ec962',  # 黄绿
    3e9: '#27ad81',    # 绿青
    7e9: '#2c728e',    # 蓝绿
    8e9: '#2c728e',    # 蓝绿
    14e9: '#440154',   # 深紫 (最大)
    # for data dup factor / slice factor (从小到大：深到浅，slice factor越小数据越稀疏用更深色)
    1: '#440154',      # 深紫 (最小，最稀疏)
    2: '#472d7b',      # 紫蓝
    4: '#3b528b',      # 蓝紫
    5: '#3b528b',      # 蓝紫
    # 5: '#2c728e',      # 蓝绿
    10: '#21918c',     # 青绿
    20: '#21918c',     # 青绿
    # 20: '#27ad81',     # 绿青
    25: '#5ec962',     # 黄绿
    50: '#aadc32',     # 黄绿色
    100: '#aadc32',    # 黄绿色
    # 100: '#fde725',    # 明黄 (最大，最密集)
    # for runs (run1 is 深紫)
    'run0': '#5ec962', # 黄绿
    'run1': '#440154', # 深紫
    'run2': '#21918c', # 青绿
    # for E
    53760: '#21918c',
    53248: '#21918c',  # Add the actual E_max value, same as 53760
    # for model type comparison
    'base': '#27ad81',    # 绿青
    'instruct': '#440154', # 深紫
}

def get_color_for_curve(curve_id):
    """
    Get color for curve_id, with fallback for unknown values
    """
    if curve_id in COLOR_MAPPING:
        return COLOR_MAPPING[curve_id]
    
    # Try to convert numpy types to standard types
    try:
        if hasattr(curve_id, 'item'):  # numpy scalar
            standard_id = curve_id.item()
            if standard_id in COLOR_MAPPING:
                return COLOR_MAPPING[standard_id]
    except:
        pass
    
    # Try converting to int (for E values)
    try:
        int_id = int(float(curve_id))
        if int_id in COLOR_MAPPING:
            return COLOR_MAPPING[int_id]
    except:
        pass
    print("Warning: plot color not found for curve_id:", curve_id, "use hash-based color")
    # Fallback: use hash to generate consistent color
    import matplotlib.pyplot as plt
    colors = plt.cm.tab10.colors
    hash_val = hash(str(curve_id)) % len(colors)
    return colors[hash_val]