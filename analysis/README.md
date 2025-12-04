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
- Existing models:
  - loglinear:
    - log(y) = -k(n) * log(x) + E0(n)
    - k & E0 lookup table
  - invexp:
    - L = (S/x)^k(n);
    - k lookup table, all curve cross same point x=S
  - powlaw:
    - L = E + (A/n)^alpha + (B/x)^beta
  - powlawmul:
    - L(N, C) = ((C0 * N^r) / C )^( k_{max} * N / (N + N0) )
    - cross point varies with N
  - postopenai:
    - L(N, C) = [(N0/N)^β + B0(N)/(C+C0(N))]^α
    - 17 params: N0, β, α (global) + B0(N), C0(N) lookup tables
    - R² > 0.995 on full dataset (0.5B-72B)
  - invexp_klinear, invexp_kquadlog, invexp_kexp: other invariants

<!-- - Adding new fitting model
  1. implement `BaseFitter`-based class
  2. import in `src/fit/models/__init__.py`: `from .loglinear_tau import LogLinearTau`
  3. use it as `--fit-model {MODEL_NAME}`
  4. carefully set bound `DEFAULT_BOUNDS` and initial value `DEFAULT_P0` for parameters, to guide the fitting search
  5. easy test using `./scripts/exp1_fit_model_test.sh` (fit + plot, skip result saving) -->

**Note on fitting quality:**
- If fitted curves are significantly off or missing in plots, re-run the fitting pipeline. CMA-ES optimization has randomness and may occasionally fail to find the optimal solution.

### Config
Important config (in config.py):
- `DEFAULT_LABELS`: defines valid column/metric names (e.g., 'C', 'E', 'ErrRate'). Add new entries here to extend available options for `--plot-x`, `--plot-curve`, `--plot-metric`, etc.


## Quick start

### Model Extrapolation with E0_72 Compensation

**Recommended workflow for 32B→72B extrapolation:**

```bash
# Step 1: Fit loglinear_kn model on 0.5B-32B data
./scripts/exp1_fits_save_up32B.sh
# Output: outputs/fits_exp1_up32B.json

# Step 2: Optimize E0_72 as compensation parameter
uv run python scripts/optimize_e072_compensation.py
# Output: outputs/fits_hybrid_kn_e072.json

# Step 3: Plot extrapolation results (72B shown with dashed line)
./scripts/plot_hybrid_extrapolation.sh
# Output: outputs/extrap_*.pdf (4 plots)

# Step 4: Plot K(N) coefficient comparison (4-subplot layout)
./scripts/plot_hybrid_coefficients.sh
# Output: outputs/holdout_N_C_raw_vs_E_k_compare_x_combined.pdf
# Layout: 2×2 grid [Base L(N,C), Base L(N,E), Instruct L(N,C), Instruct L(N,E)]
# Each subplot shows: 0.5-32B fit points, K(N) curve (dashed), Actual 72B point (star)

# Step 5: Generate K(N) and E0(N) value tables
uv run python scripts/generate_kn_e0_tables.py > outputs/kn_e0_tables.txt
# Output: LaTeX tables with K(N) and E0(N) values for all model sizes (0.5B-72B)
```

**Key features:**
- **K(N) function**: k(N) = k_max × N/(N+N0) fitted on 0.5B-32B data
- **E0_72 compensation**: Optimized separately to minimize 72B prediction error
- **Actual 72B comparison**: Star markers show ground truth K(72B) from full dataset fit
- **Extrapolation quality**: R² = 0.81-0.88 on 72B with only 1 compensated parameter

### Per-Model Analysis (K, E0, R² Tables)

Generate LaTeX tables with per-model K(N), E0(N), R² values (requires steps 1-2 above):
```bash
./scripts/analyze_per_model.sh
# Output: outputs/per_model_compact_tables.tex (ready for paper)
```

### Standard Fitting (Full Dataset)

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

Cross Domain (8-subplot grid: 2×4 layout with all evaluations)
```bash
./scripts/eval_final_vs_model_size.sh
# Output: crossdomain_base_ErrRate_vs_N.pdf, crossdomain_instruct_ErrRate_vs_N.pdf
```

Cross Domain (one line per domain)
```bash
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source base -N 72e9 --warmup-clip-to 10 --eval-group in_domain
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source base -N 72e9 --warmup-clip-to 10 --eval-group out_of_domain
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source instruct -N 72e9 --warmup-clip-to 10 --eval-group in_domain
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source instruct -N 72e9 --warmup-clip-to 10 --eval-group out_of_domain
uv run -m src.run.eval_curves_by_model --help
```

Plot All Dataset Evaluations (deprecated, one subplot per domain)
```bash
uv run -m src.run.eval_subplots --data-source base --x-columns E --metrics ErrRate --warmup-clip-to 10
uv run -m src.run.eval_subplots --data-source instruct --x-columns E --metrics ErrRate --warmup-clip-to 10

uv run -m src.run.eval_subplots --data-source base --x-columns C_raw --metrics ErrRate --warmup-clip-to 10
uv run -m src.run.eval_subplots --data-source instruct --x-columns C_raw --metrics ErrRate --warmup-clip-to 10

uv run -m src.run.eval_subplots --help
```

Single side scaling law
```bash
./scripts/exp1_plot_fit_singleside.sh
```