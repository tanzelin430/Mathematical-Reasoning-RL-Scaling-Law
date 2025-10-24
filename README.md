# LLM-RL-Scaling-Law-Analysis

## Usage

### scaling_analysis.py

- We use `scaling_analysis.py` as the all-in-one main analysis entrance.
It covers fitting, plotting, smoothing (multi-segment fitting), param analysis.

```bash
uv run -m src.scaling_analysis --help
```

- basic 5-dimention plot:
  - data source: 
    - "base", "instruct", "llama-base", "llama-instruct"
  - x: 
    - "C": Compute
    - "E": Data size (step * batch size)
    - "T": Token
  - eval
    - holdout_score, gsm8k, response_length etc.
    - checkout `config.TEST_EVALS`
  - metric:
    - R: raw reward score
    - ErrRate: 1-R
    - DeltaR: R_i - R_{base_step}
    - DeltaErrRate: ErrRate_i - ErrRate_{base_step}
  - curve:
    - "N": draw a curve for each model size
    - "slice_factor": data duplication factor


- simple demo can be found here
```bash
uv run -m src.demo
```
### Data prepare
**Make sure to extract the correct csv format before proceed**

```bash
# preprocess data to csv
uv run -m src.extract_csv
```

Adding data source:
- edit `extract_csv.py` - `main()`
- add to `config.py` - `CSV_MAP`

Note: 
- in csv: *rollout_n* column is only for GRPO experiments


### Fitting
- Adding new fitting model
  - implement `BaseFitter`-based class
  - import in `src/fit/models/__init__.py`: `from .loglinear_tau import LogLinearTau`
  - use it as `--fit-model {MODEL_NAME}`

### Config
Important config (in config.py):
- `DEFAULT_X_LABELS` `DEFAULT_Y_LABELS`: control default figure labels


## Quick start

Compute(C), Datasize(E) vs TestLoss:
**Note: to change fit model:**
- update `exp1_fits_save.sh` `--fit-model loglinear` to `invexp` etc.

```bash
# Fitting for (C & E * base & instruct)
./scripts/exp1_fits_save.sh
# [Figure 1, 2, 3] Fitting - 4 figures: (C & E * base & instruct)
./scripts/exp1_plot_fit.sh
# [optional] Smooth - 4 figures: (C & E * base & instruct)
./scripts/exp1_plot_smooth.sh
```

k(curve) / E0(curve) analysis:
- `--fit-param-plot-schema` choices: 
  - 'single': naive plot for each param
  - 'compare-source':
    - plot files: L(N,C) & L(N,D) 
    - subplot: k & E0
    - curves: base vs instruct
  - 'compare-x': 
    - plot files: base & instruct
    - subplot: k & E0 
    - curves: L(N,C) vs L(N,D)
  - 'table': **Latex** table, single-column
  - 'table-compact': **Latex** table compact version, for 2 columns

```bash
# Load fitting results and plot parameters:

# plot files: L(N,C) & L(N,D) - subplot: k & E0 - curves: base vs instruct
uv run -m src.scaling_analysis \
  --fit-load outputs/fits_exp1.json \
  --fit-param-plot-schema compare-source\
  --plot-x-scale log \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-legend-loc "upper left"\
  --scatter-alpha 1 \
  --scatter-size 100 \
  --scatter-marker o \
  --output-prefix fit_param_

# latex tables for k & E0
uv run -m src.scaling_analysis \
  --fit-load outputs/fits_exp1.json \
  --fit-param-plot-schema table-compact\
  --plot-x-scale log \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-legend-loc "upper left"\
  --scatter-alpha 1 \
  --scatter-size 100 \
  --scatter-marker o \
  --output-prefix fit_param_
```

Response Length
```bash
# 
# Response length - Datasize (including Figure 4b)
./scripts/response_length_E.sh
# [optional] Preview TestLoss - Response length
./scripts/response_length_testloss.sh
```

Data reuse

- data prepare: use exp1 7B as tau=1 data
```bash
cp -r data/Experiment1_Base/Experiment1_Base_run0/7B data/experiment2_base/7B/run_1
cp -r data/Experiment1_Instruct/Experiment1_Instruct_run2/7B data/experiment2_instruct/7B/run_1
# extract to csv
uv run -m src.extract_csv
```
- plotting
```bash
# test loss vs steps - smooth curve - (base & instruct)
./scripts/exp2_smooth.sh
# test loss vs steps - fitting L(\tau, steps) (log linear) - (base & instruct)
./scripts/exp2_fit_loglinear.sh

# use --fit-param-plot-schema to plot k(\tau) / E0(\tau)
```

- Data reuse experiment setup blocks
```bash
# experiment setup blocks
uv run -m src.run.exp2_setup_rectangle
```

GRPO
```bash
./scripts/grpo.sh
```

Cross Domain (one line per domain)
```bash
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source base -N 72e9 --eval-group in_domain
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source base -N 72e9 --eval-group out_of_domain
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source instruct -N 72e9 --eval-group in_domain
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source instruct -N 72e9 --eval-group out_of_domain
uv run -m src.run.eval_curves_by_model --help
```

Plot All Dataset Evaluations (deprecated, one subplot per domain)
```bash
uv run -m src.run.eval_subplots --data-source base --x-columns E --metrics ErrRate --warmup-clip 10
uv run -m src.run.eval_subplots --data-source instruct --x-columns E --metrics ErrRate --warmup-clip 10

uv run -m src.run.eval_subplots --data-source base --x-columns C_raw --metrics ErrRate --warmup-clip 10
uv run -m src.run.eval_subplots --data-source instruct --x-columns C_raw --metrics ErrRate --warmup-clip 10
```

Single side scaling law (Deprecated)

<!-- 
```bash
# demo plot, 5-dimention control (model type, x, eval, metric, curve), support multiple plot: etc. ["C", "E"] x ["R", "ErrRate"]
uv run -m src.run.demo

# to plot multiple figures with fitting lines, same as above (dual logistic with k(N)=log(1-xxx) version)
uv run -m src.scaling_analysis
``` -->
