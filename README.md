# LLM-RL-Scaling-Law-Analysis

## Quick start
```bash
$ uv run python extract_csv.py
$ uv run python run.py
```

Key switch:
- `PLOT_BASIC_CURVES: bool` : 
  - True for plotting basic <Reward/ErrRate - Dataset Metric - Compute/Datasize/Token> figures
  - False for Intrinsic Performance computations
- `HOLDOUT: bool`:
  - True for holdout only test set
  - False for multi-dataset eval plotting
