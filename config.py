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
WARMUP_CLIPPING_FACTOR_FOR_RAW = 15/100  # on raw data
WARMUP_CLIPPING_FACTOR_FOR_SMOOTH = 0/100  # on smoothed data, [based on clipped raw]

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
    'val/test_score/aimeamc2023': {'file_str': 'aimeamc2023', 'plot_str': 'AIME 2023'},
    'val/test_score/aime2024': {'file_str': 'aime2024', 'plot_str': 'AIME 2024'},
    # 'val/test_score/math__deepscaler_preview': {'file_str': 'math__deepscaler_preview', 'plot_str': 'DeepScaler Preview'},
    # 'val/test_score/math__merged_deduped_dapo_or1_dataset': {'file_str': 'merged_deduped_dapo', 'plot_str': 'Merged Deduped Dapo OR1 Dataset'},
}
MULTI_FIGURE_COLUMNS = 2 # note: if total > figure_columns, [row, col] -> [i]
MULTI_FIGURE_SIZE=(10, 10)

TOTAL_EVALS = len(TEST_EVALS.keys())


DEFAULT_Y_LABELS = {
    'R': "Reward", 
    'ErrRate': "Error Rate", 
    'DeltaReward': "Improvement",
    'DeltaErrRate': "Delta Error Rate"
}

DEFAULT_X_LABELS = {
    "T": "Tokens",
    "C": "Compute (FLOPs)",
    "E": "Data Size"
}

DEBUG = False