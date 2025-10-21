# LLM-RL-Scaling-Law-Analysis

## Quick start

```bash
# preprocess data to csv
uv run -m src.extract_csv
```

Compute(C), Datasize(E) vs TestLoss:
```bash
# fitting - 4 figures: (C & E * base & instruct)
./scripts/exp1_fit.sh

# optional: smooth - 4 figures: (C & E * base & instruct)
./scripts/exp1_smooth.sh
```

k - curve / E0 - curve:
```bash
# k(curve) - Base & Instruct
# E0(curve) - Base & Instruct
# curve = N / Tau

# plot k(N) on L(N, C_raw)
uv run -m src.run.xhyu_k_e0_scatters_baseinstruct --plot-enable k E0 --data-sources base instruct --curve-column N --fitting-type C_raw --warmup-clip 10

# plot appendix figure: L(N,D) vs L(N,C) k & E0
uv run -m src.run.xhyu_k_e0_scatters --model-type base --warmup-clip 10
uv run -m src.run.xhyu_k_e0_scatters --model-type instruct --warmup-clip 10
```

Data reuse
```bash
# test loss vs steps - smooth curve - (base & instruct)
./scripts/exp2_smooth.sh

# test loss vs steps - fitting (log linear) - (base & instruct)
./scripts/exp2_fit_loglinear.sh

# plot k(\tau) on L(\tau, steps)
# k & E0
uv run -m src.run.xhyu_k_e0_scatters_baseinstruct --plot-enable k E0 --data-sources exp2-base exp2-instruct --curve-column Tau --fitting-type step --warmup-clip 10
# k
uv run -m src.run.xhyu_k_e0_scatters_baseinstruct --plot-enable k --data-sources exp2-base exp2-instruct --curve-column Tau --fitting-type step --warmup-clip 10
# E0
uv run -m src.run.xhyu_k_e0_scatters_baseinstruct --plot-enable E0 --data-sources exp2-base exp2-instruct --curve-column Tau --fitting-type step --warmup-clip 10

# experiment setup blocks
uv run -m src.run.exp2_setup_rectangle
```

notes: remember to edit this when adding new columns
```python
df_mean = (
    df.groupby([curve_column, 'step'], as_index=False)
      .agg(N=('N', 'first'), Tau=('Tau', 'first'), C=('C', 'first'), C_raw=('C_raw', 'first'), E=('E', 'first'), ErrRate=('ErrRate', 'mean'), ImprovementRate=('ImprovementRate', 'mean'))
)
```

Response Length
```bash
./scripts/response_length_E.sh
./scripts/response_length_testloss.sh

# k & E0 together
uv run -m src.run.xhyu_k_e0_scatters_baseinstruct --plot-enable k E0 --data-sources base instruct --curve-column N --fitting-type response_length --warmup-clip 10
# k
uv run -m src.run.xhyu_k_e0_scatters_baseinstruct --plot-enable k --data-sources base instruct --curve-column N --fitting-type response_length --warmup-clip 10 
# E0
uv run -m src.run.xhyu_k_e0_scatters_baseinstruct --plot-enable E0 --data-sources base instruct --curve-column N --fitting-type response_length --warmup-clip 10 
```

Single side scaling law
```bash
# L(D)
uv run -m src.run.plot_multi_fit_simple_curve_fit_fixN

# L(N)
uv run -m src.run.plot_multi_fit_simple_curve_fit_fixE
```

GRPO
```bash
./scripts/grpo.sh
```

Cross Domain (one line per domain)
```bash
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source base -N 14e9 --eval-group in_domain
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source base -N 14e9 --eval-group out_of_domain
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source instruct -N 14e9 --eval-group in_domain
uv run -m src.run.eval_curves_by_model --x-columns C --metrics ErrRate --data-source instruct -N 14e9 --eval-group out_of_domain
uv run -m src.run.eval_curves_by_model --help
```

Plot All Dataset Evaluations (deprecated, one subplot per domain)
```bash
uv run -m src.run.eval_subplots --data-source base --x-columns E --metrics ErrRate --warmup-clip 10
uv run -m src.run.eval_subplots --data-source instruct --x-columns E --metrics ErrRate --warmup-clip 10

uv run -m src.run.eval_subplots --data-source base --x-columns C_raw --metrics ErrRate --warmup-clip 10
uv run -m src.run.eval_subplots --data-source instruct --x-columns C_raw --metrics ErrRate --warmup-clip 10
```

<!-- 
```bash
# demo plot, 5-dimention control (model type, x, eval, metric, curve), support multiple plot: etc. ["C", "E"] x ["R", "ErrRate"]
uv run -m src.run.demo

# to plot multiple figures with fitting lines, same as above (dual logistic with k(N)=log(1-xxx) version)
uv run -m src.run.plot_multi_fit
``` -->

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
- `DEFAULT_X_LABELS` `DEFAULT_Y_LABELS`: control default figure labels


Adding data source:
- edit `extract_csv.py` - `main()`
- add to `config.py` - `CSV_MAP`

Note: 
- reuse exp1 7B run0 as tau=1 data
  ```bash
  cp -r data/Experiment1_Base/Experiment1_Base_run0/7B data/experiment2_base/7B/run_1
  cp -r data/Experiment1_Instruct/Experiment1_Instruct_run0/7B data/experiment2_instruct/7B/run_1
  ```
- *rollout_n* column is only for grpo experiments