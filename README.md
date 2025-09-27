# LLM-RL-Scaling-Law-Analysis

## Quick start

- view the code briefly before execute, it may contain config info

```bash
uv run python extract_csv.py
```
```bash
# now support cmd line interface, with config files available for pipeline
uv run python run_plot_multi_fit.py --config-file example_batch_config.json
```
```bash
# to plot single figure, with 5 dimention control (model type, x, eval, metric, curve)
uv run python run_plot_single.py

# to plot multiple figures at a time, very handy. etc. ["C", "E"] x ["R", "ErrRate"]
uv run python run_plot_multi.py

# to plot multiple figures with fitting lines, same as above (dual logistic with k(N)=log(1-xxx) version)
uv run python run_plot_multi_fit.py

# to plot data duplication curves
uv run python run_plot_single_slicefactor.py

# other run_... scripts are not ready to use, those mainly for testing purpose.
```

Cross Domain
```bash
uv run python run_plot_multi-subplot_oodid.py --x-columns C --metrics ErrRate --data-source base --eval-group in_domain
uv run python run_plot_multi-subplot_oodid.py --x-columns C --metrics ErrRate --data-source base --eval-group out_of_domain
uv run python run_plot_multi-subplot_oodid.py --x-columns C --metrics ErrRate --data-source instruct --eval-group in_domain
uv run python run_plot_multi-subplot_oodid.py --x-columns C --metrics ErrRate --data-source instruct --eval-group out_of_domain
uv run python run_plot_multi-subplot_oodid.py --help
```

k - curve / E0 - curve:
```bash
# k(curve) - Base & Instruct
# E0(curve) - Base & Instruct
# curve = N / Tau

# plot k(N) on L(N, C_raw)
uv run python run_xhyu_k_e0_scatters_baseinstruct.py --plot-enable k E0 --data-sources base instruct --curve-column N --fitting-type C_raw --warmup-clip-num 10

# plot appendix figure: L(N,D) vs L(N,C) k & E0
uv run python run_xhyu_k_e0_scatters.py --model-type base --warmup-clip-num 10
uv run python run_xhyu_k_e0_scatters.py --model-type instruct --warmup-clip-num 10

# plot single k(N), E0(N):
uv run python run_plot_multi_fit.py --config-file exp2_k_e0.json
```

Data reuse
```bash
# test loss vs steps - smooth curve version
uv run python run_plot_multi_fit.py --config-file params/datareuse_tau_smooth.json

# test loss vs steps - fit line version
uv run python run_plot_multi_fit.py --config-file params/datareuse_tau_fit.json

# plot k(\tau) on L(\tau, steps)
# k & E0
uv run python run_xhyu_k_e0_scatters_baseinstruct.py --plot-enable k E0 --data-sources exp2-base exp2-instruct --curve-column Tau --fitting-type step --warmup-clip-num 10
# k
uv run python run_xhyu_k_e0_scatters_baseinstruct.py --plot-enable k --data-sources exp2-base exp2-instruct --curve-column Tau --fitting-type step --warmup-clip-num 10
# E0
uv run python run_xhyu_k_e0_scatters_baseinstruct.py --plot-enable E0 --data-sources exp2-base exp2-instruct --curve-column Tau --fitting-type step --warmup-clip-num 10

# experiment setup blocks
uv run python run_rectangle_test.py
```
notes: remember to edit this when adding new columns
```python
df_mean = (
    df.groupby([curve_column, 'step'], as_index=False)
      .agg(N=('N', 'first'), Tau=('Tau', 'first'), C=('C', 'first'), C_raw=('C_raw', 'first'), E=('E', 'first'), ErrRate=('ErrRate', 'mean'), ImprovementRate=('ImprovementRate', 'mean'))
)
```

notes: when update experiment2, remember to copy run_0 result from experiment1 7B, to exp2 csv, can replace slice_factor from 0 to 1

Response Length
```bash
uv run python run_plot_multi_fit.py --config-file params/response_length_E.json
uv run python run_plot_multi_fit.py --config-file params/response_length_testloss.json

# k & E0 together
uv run python run_xhyu_k_e0_scatters_baseinstruct.py --plot-enable k E0 --data-sources base instruct --curve-column N --fitting-type response_length --warmup-clip-num 10
# k
uv run python run_xhyu_k_e0_scatters_baseinstruct.py --plot-enable k --data-sources base instruct --curve-column N --fitting-type response_length --warmup-clip-num 10 
# E0
uv run python run_xhyu_k_e0_scatters_baseinstruct.py --plot-enable E0 --data-sources base instruct --curve-column N --fitting-type response_length --warmup-clip-num 10 
```

Single side scaling law
```bash
# L(D)
uv run python run_plot_multi_fit_simple_curve_fit_fixN.py

# L(N)
uv run python run_plot_multi_fit_simple_curve_fit_fixE.py
```

GRPO
```bash
uv run python run_plot_multi_fit.py --config-file params/grpo.json
```

Plot single N-C-ErrRate with smooth (not fit):
```bash
uv run python run_plot_multi_fit.py --config-file test_NC_smooth.json
```

All Evals
```bash
uv run python run_plot_multi-subplot.py --data-source base --x-columns E --metrics ErrRate --warmup-clip-num 10
uv run python run_plot_multi-subplot.py --data-source instruct --x-columns E --metrics ErrRate --warmup-clip-num 10

uv run python run_plot_multi-subplot.py --data-source base --x-columns C_raw --metrics ErrRate --warmup-clip-num 10
uv run python run_plot_multi-subplot.py --data-source instruct --x-columns C_raw --metrics ErrRate --warmup-clip-num 10
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


Adding data source:
- edit `extract_csv.py` - `main()`
- add to `config.py` - `CSV_MAP`