from pathlib import Path
import os
import sys
# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION - Edit these variables to customize the run
# =============================================================================

# Data source configuration - use absolute paths based on script location
PROJECT_DIR = Path(__file__).parent.parent.parent
OUTPUT_BASE_DIR = PROJECT_DIR / "outputs"  # Base output directory for PNG plots
SAMPLE_SIZE_PER_STEP = 512
BUILD_I_ON_SMOOTHED = True

# HOLDOUT=True,
# Test evals to process (from the CSV columns)
DEFAULT_TEST_EVAL = 'holdout_score'
DEFAULT_FIGURE_PREFIX = 'holdout'
# DEFAULT_FIGURE_COLUMNS = 1 # note: if total > figure_columns, [row, col] -> [i]
# DEFAULT_FIGURE_SIZE=(5, 5)

# Test evals to process (from the CSV columns)
TEST_EVALS = {
    'holdout_score': {'file_str': 'holdout', 'plot_str': 'Holdout Validation'},
    'response_length': {'file_str': 'response_length', 'plot_str': 'Response Length'},
    # 'overall_pass1': {'file_str': 'overall_pass1', 'plot_str': 'Overall@Pass1'},
    'val/test_score/openai/gsm8k': {'file_str': 'gsm8k', 'plot_str': 'GSM8K'},
    'val/test_score/codegen__humaneval': {'file_str': 'codegen__humaneval', 'plot_str': 'CodeGen - HumanEval'},
    'val/test_score/stem__supergpqa': {'file_str': 'stem__supergpqa', 'plot_str': 'SuperGPQA'},
    'val/test_score/math__math': {'file_str': 'math__math', 'plot_str': 'MATH-500'},
    'val/test_score/logic__zebra_puzzle_dataset': {'file_str': 'logic__zebra_puzzle_dataset', 'plot_str': 'Logic - Zebra Puzzle'},
    'val/test_score/aimeamc2023': {'file_str': 'aimeamc2023', 'plot_str': 'AMC2023'},
    'val/test_score/aime2024': {'file_str': 'aime2024', 'plot_str': 'AIME2024'},
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
    "Tau": "Data Reuse",
    "slice_factor": "Data Reuse",
    "step": "Steps",
    "rollout_n": "ρ",

    'base': 'Base', 
    'instruct': 'Instruct',
    'exp2-base': '7B-Base',
    'exp2-instruct': '7B-Instruct',

    'response_length': 'Response Length',
}

DEFAULT_SHORT_NAME = {
    "C_raw": "C",
    "E": "D",
    "slice_factor": "Tau",
    "N": "N",
    "Tau": "τ",
    "rollout_n": "ρ",
    'response_length': 'response',
}

# 列名映射 - 标准化数据列名
COLUMN_RENAME_MAP = {
    'model_params': 'N',
    'cumulative_flops': 'C_raw',
    'runid': 'runid',
    'step': 'step',
    'tokens': 'tokens',
    'cumulative_tokens': 'T',
    'slice_factor': 'Tau',
}

DEBUG = False

CSV_INSTRUCT_RUNS = [
        # PROJECT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run0.csv" ,
        PROJECT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run1.csv" ,
        PROJECT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run2.csv" ,
    ]

CSV_BASE_RUNS = [
    PROJECT_DIR / "csv" / "scaling_law_data_experiment1_base_run0.csv" ,
    PROJECT_DIR / "csv" / "scaling_law_data_experiment1_base_run1.csv" ,
    PROJECT_DIR / "csv" / "scaling_law_data_experiment1_base_run2.csv" ,
]

CSV_LLAMA_BASE_RUNS = [
    PROJECT_DIR / "csv" / "scaling_law_data_experiment-llama-base.csv" ,
    PROJECT_DIR / "csv" / "scaling_law_data_experiment-llama-base-run0.csv" ,
]

CSV_LLAMA_INSTRUCT_RUNS = [
    PROJECT_DIR / "csv" / "scaling_law_data_experiment-llama-instruct.csv" ,
]

CSV_EXPERIMENT2_BASE_RUNS = [
    PROJECT_DIR / "csv" / "scaling_law_data_experiment2_base.csv" ,
]

CSV_EXPERIMENT2_INSTRUCT_RUNS = [
    PROJECT_DIR / "csv" / "scaling_law_data_experiment2_instruct.csv" ,
]

CSV_GRPO_BASE_RUNS = [
    PROJECT_DIR / "csv" / "scaling_law_data_grpo_base.csv" ,
]

CSV_GRPO_INSTRUCT_RUNS = [
    PROJECT_DIR / "csv" / "scaling_law_data_grpo_instruct.csv" ,
]

CSV_MAP = {
    "base": CSV_BASE_RUNS,
    "instruct": CSV_INSTRUCT_RUNS,
    "llama-base": CSV_LLAMA_BASE_RUNS,
    "llama-instruct": CSV_LLAMA_INSTRUCT_RUNS,
    "exp2-base": CSV_EXPERIMENT2_BASE_RUNS,
    "exp2-instruct": CSV_EXPERIMENT2_INSTRUCT_RUNS,
    "grpo-base": CSV_GRPO_BASE_RUNS,
    "grpo-instruct": CSV_GRPO_INSTRUCT_RUNS,
}

# Physical dimensions for merge_duplicate_steps
# Defines which columns uniquely identify experimental conditions for each data source
# Format: [primary_dimension, 'step'] where primary_dimension is the experimental variable
PHYSICAL_DIMENSIONS = {
    "base": ['N', 'step'],                    # Experiment 1: varying model size N
    "instruct": ['N', 'step'],                # Experiment 1: varying model size N
    "llama-base": ['N', 'step'],              # Llama: varying model size N
    "llama-instruct": ['N', 'step'],          # Llama: varying model size N
    "exp2-base": ['slice_factor', 'step'],    # Experiment 2: fixed N=7B, varying data reuse slice_factor
    "exp2-instruct": ['slice_factor', 'step'],# Experiment 2: fixed N=7B, varying data reuse slice_factor
    "grpo-base": ['rollout_n', 'step'],       # GRPO: fixed N=7B, varying rollout strategy
    "grpo-instruct": ['rollout_n', 'step'],   # GRPO: fixed N=7B, varying rollout strategy
}

def get_physical_dimensions(data_source: str):
    if data_source not in PHYSICAL_DIMENSIONS:
        raise ValueError(
            f"Physical dimensions not configured for data_source: '{data_source}'. "
        )
    return PHYSICAL_DIMENSIONS[data_source]

# =============================================================================
# COLOR MAPPING - 统一的渐变配色方案
# =============================================================================

COLOR_MAPPING = {
    # for model size (从小到大：浅到深，平滑渐变从黄绿到深紫)
    0.5e9: '#fde725',  # 明黄 (最小，最浅)
    1e9: '#fde725',    # 明黄
    1.5e9: '#aadc32',  # 黄绿
    3e9: '#5ec962',    # 绿色
    7e9: '#27ad81',    # 绿青
    8e9: '#27ad81',    # 绿青
    14e9: '#2c728e',   # 蓝绿
    32e9: '#5a67a8',   # 蓝紫
    72e9: '#440154',   # 深紫 (最大，最深)
    # for data dup factor / slice factor (从小到大：深到浅，slice factor越小数据越稀疏用更深色)
    # for data dup factor / slice factor (彩虹配色：从紫到红)
    1: '#FF4500',      # 橙红 (最小，最稀疏)
    2: '#FFA500',      # 橙色
    # 4: '#FFD700',      # 金黄
    5: '#ADFF2F',      # 黄绿
    # 5: '#00FF00',      # 绿色
    10: '#00FF00',     # 绿色
    20: '#00FF00',     # 绿色
    # 20: '#00CED1',     # 暗青
    # 20: '#FFD700',     # 金黄
    25: '#00BFFF',     # 天蓝
    50: '#4169E1',     # 皇家蓝
    100: '#8A2BE2',    # 蓝紫

    # 1: '#8A2BE2',      # 蓝紫 (最小，最稀疏)
    # 2: '#4169E1',      # 皇家蓝
    # 4: '#00BFFF',      # 天蓝
    # 5: '#00CED1',      # 暗青
    # # 5: '#00FF00',      # 绿色
    # 10: '#00FF00',     # 绿色
    # 20: '#ADFF2F',     # 黄绿
    # # 20: '#FFD700',     # 金黄
    # 25: '#FFD700',     # 金黄
    # 50: '#FFA500',     # 橙色
    # 100: '#FF4500',    # 橙红
    
    # for GRPO rollout strategies (从小到大: 橙色系渐变)
    'rho4': '#FF6B35',   # 橙红 (最小rollout)
    'rho8': '#F7931E',   # 橙色
    'rho16': '#FFD23F',  # 金黄
    'rho32': '#06D6A0',  # 青绿 (最大rollout)

    'exp2-base': '#F7931E',   # 橙色
    'exp2-instruct': '#06D6A0',  # 青绿

    # 100: '#fde725',    # 明黄 (最大，最密集)
    # for runs (run1 is 深紫)
    'run0': '#5ec962', # 黄绿
    'run1': '#440154', # 深紫
    'run2': '#21918c', # 青绿
    # for model type comparison
    'base': '#27ad81',    # 绿青
    'instruct': '#440154', # 深紫
    'C_raw': '#27ad81',    # 绿青
    'E': '#440154', # 深紫
    'k': 'blue',    # 绿青
    'E0': 'red', # 深紫
    'S': 'green', # 深紫

    # # for evaluations (现代配色方案)
    # # In-domain: 暖色调渐变 (红橙黄绿青)
    # 'holdout_score': '#E74C3C',  # 现代红色
    # 'val/test_score/openai/gsm8k': '#F39C12',  # 现代橙色
    # 'val/test_score/math__math': '#F1C40F',  # 现代黄色
    # 'val/test_score/aime2024': '#27AE60',  # 现代绿色
    # 'val/test_score/aimeamc2023': '#16A085',  # 现代青色
    # # Out-of-domain: 冷色调渐变 (深蓝到浅蓝系列)
    # 'val/test_score/stem__supergpqa': '#2E86AB',  # 深海蓝
    # 'val/test_score/codegen__humaneval': '#A23B72',  # 深紫红
    # 'val/test_score/logic__zebra_puzzle_dataset': '#1B4F72',  # 深靛蓝 (远离红色)
    
    # 备用冷色调颜色选项 (更多同色系差异化选择)
    'cold_option_1': '#154360',  # 深蓝
    'cold_option_2': '#1F618D',  # 钢蓝  
    'cold_option_3': '#2874A6',  # 中蓝
    'cold_option_4': '#3498DB',  # 亮蓝
    'cold_option_5': '#5DADE2',  # 浅蓝
    'cold_option_6': '#85C1E9',  # 天蓝
    'cold_option_7': '#2E4057',  # 深青灰
    'cold_option_8': '#048A81',  # 深青绿
    'cold_option_9': '#54C6EB',  # 青蓝
    'cold_option_10': '#006BA6', # 皇家蓝
    
    # Rainbow配色方案备选 (保持暖冷区分但更鲜艳)
    # In-domain: Rainbow暖色段 (红橙黄绿)
    'holdout_score': '#FF1744',  # 鲜红
    'val/test_score/aime2024': '#FF6D00',  # 鲜橙
    'val/test_score/aimeamc2023': '#FFD600',  # 鲜黄
    'val/test_score/math__math': '#00E676',  # 鲜绿
    'val/test_score/openai/gsm8k': '#00BCD4',  # 青色
    # Out-of-domain: Rainbow冷色段 (蓝紫靛)
    'val/test_score/logic__zebra_puzzle_dataset': '#3F51B5',  # 鲜蓝
    'val/test_score/stem__supergpqa': '#9C27B0',  # 靛蓝
    'val/test_score/codegen__humaneval': '#2196F3',  # 鲜紫
}

DEFAULT_MARKERS = {'base': 'o', 'instruct': 's', 'exp2-base': 'o', 'exp2-instruct': 's'}
    

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