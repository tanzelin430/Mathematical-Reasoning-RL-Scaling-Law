# LLM-RL-Scaling-Law-Analysis

## Quick start

- view the code briefly before execute, it may contain config info

```bash
$ uv run python extract_csv.py
```
```bash
# now support cmd line interface, with config files available for pipeline
$ uv run python run_plot_multi_fit.py --config-file example_batch_config.json
```
```bash
# to plot single figure, with 5 dimention control (model type, x, eval, metric, curve)
$ uv run python run_plot_single.py

# to plot multiple figures at a time, very handy. etc. ["C", "E"] x ["R", "ErrRate"]
$ uv run python run_plot_multi.py

# to plot multiple figures with fitting lines, same as above (dual logistic with k(N)=log(1-xxx) version)
$ uv run python run_plot_multi_fit.py

# to plot data duplication curves
$ uv run python run_plot_single_slicefactor.py

# other run_... scripts are not ready to use, those mainly for testing purpose.
```

5-dimention control:
- model type: 
  - "base", "instruct", "llama-base", "llama-instruct"
- x: 
  - "C": Compute
  - "E": Data size (step * batch size)
  - "T": Token
- eval
  - holdout_score
  - gsm8k, etc.
  - checkout config.TEST_EVALS
- metric:
  - R: raw reward score
  - ErrRate: 1-R
  - DeltaR: R_i - R_{base_step}
  - DeltaErrRate: ErrRate_i - ErrRate_{base_step}
- curve by:
  - "N": draw a curve for each model size
  - "slice_factor": data duplication factor

Important config (in config.py):
- `WARMUP_CLIPPING_FACTOR_FOR_RAW`: data clip ratio
- `DEFAULT_X_LABELS` `DEFAULT_Y_LABELS`: control default figure labels