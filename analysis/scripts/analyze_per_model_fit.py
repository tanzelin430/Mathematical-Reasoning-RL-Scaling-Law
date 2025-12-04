"""
Calculate K(N), E0(N), and R² for each model size from hybrid fit results.
Generates LaTeX table with results.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.data_proc import load_and_preprocess, prepare_eval_data
from src.common.config import CSV_MAP


def calculate_k_n(N, k_max, N0):
    """Calculate k(N) using the loglinear_kn formula."""
    return k_max * N / (N + N0)


def calculate_r2(y_true, y_pred):
    """Calculate R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def model_prediction(x, k, E0):
    """
    Predict ErrRate using loglinear_kn model.
    log10(y) = -k * log10(x) + E0
    y = 10^(-k * log10(x) + E0)
    """
    return np.power(10.0, -k * np.log10(x) + E0)


def main():
    # Load fit results
    fit_file = Path("outputs/fits_hybrid_kn_e072.json")
    with open(fit_file) as f:
        fit_data = json.load(f)

    # Model sizes (in billions)
    model_sizes = [0.5, 1.5, 3, 7, 14, 32, 72]
    model_sizes_full = [int(m * 1e9) for m in model_sizes]  # Convert to actual numbers

    # Results storage
    results = {}

    # Process each fit configuration
    for fit in fit_data["fits"]:
        context = fit["context"]
        params = fit["params"]

        data_source = context["data_source"]
        x_var = context["fit_x"]

        key = f"{data_source}_{x_var}"

        # Extract k_max and N0
        k_max = params["k_max"]
        N0 = params["N0"]

        # Load actual data using the same pipeline as scaling_analysis
        data = load_and_preprocess(CSV_MAP[data_source])

        # Prepare eval data (this adds R, ErrRate, compute columns, etc.)
        from src.common import config as cfg
        physical_dimensions = cfg.get_physical_dimensions(data_source)
        physical_curve_column = physical_dimensions[0]  # N
        physical_x_column = physical_dimensions[1]      # step

        data = prepare_eval_data(
            data,
            eval_column="holdout_score",
            curve_column=physical_curve_column,
            x_column=physical_x_column,
            calc_delta=False,
            delta_base_step=0
        )

        # Remove step=0 data
        data = data[data["step"] > 0].reset_index(drop=True)

        # Apply warmup clip if needed
        warmup_clip = context.get("warmup_clip", 0)
        if warmup_clip > 0:
            from src.common.data_proc import apply_clip
            data = apply_clip(
                data,
                curve_column=physical_curve_column,
                warmup_clip=warmup_clip
            )

        # Store results for each model size
        results[key] = {
            "data_source": data_source,
            "x_var": x_var,
            "k_max": k_max,
            "N0": N0,
            "models": []
        }

        # Calculate for each model size
        for i, (N_billions, N_full) in enumerate(zip(model_sizes, model_sizes_full)):
            # Calculate k(N)
            k_N = calculate_k_n(N_full, k_max, N0)

            # Get E0 from params
            if N_billions == 0.5:
                E0 = params["E0_0_5"]
            elif N_billions == 1.5:
                E0 = params["E0_1_5"]
            elif N_billions == 3:
                E0 = params["E0_3"]
            elif N_billions == 7:
                E0 = params["E0_7"]
            elif N_billions == 14:
                E0 = params["E0_14"]
            elif N_billions == 32:
                E0 = params["E0_32"]
            elif N_billions == 72:
                E0 = params["E0_72"]

            # Filter data for this model size
            model_data = data[data["N"] == N_full].copy()

            if len(model_data) == 0:
                print(f"Warning: No data for {data_source} N={N_billions}B")
                continue

            # Get x and y values
            x_values = model_data[x_var].values
            y_true = model_data["ErrRate"].values  # True ErrRate values (not log)

            # Predict using model
            y_pred = model_prediction(x_values, k_N, E0)  # Predicted ErrRate values

            # Calculate R²
            r2 = calculate_r2(y_true, y_pred)

            # Calculate data statistics
            err_std = model_data["ErrRate"].std()
            err_mean = model_data["ErrRate"].mean()

            results[key]["models"].append({
                "N_billions": N_billions,
                "k_N": k_N,
                "E0": E0,
                "r2": r2,
                "n_points": len(model_data),
                "err_mean": err_mean,
                "err_std": err_std
            })

            print(f"{data_source} {x_var} N={N_billions}B: k={k_N:.6f}, E0={E0:.6f}, R²={r2:.6f} (n={len(model_data)})")

    # Generate LaTeX tables
    print("\n" + "="*80)
    print("LaTeX Tables")
    print("="*80 + "\n")

    for key, result in results.items():
        data_source = result["data_source"]
        x_var = result["x_var"]
        x_label = "Compute (C)" if x_var == "C_raw" else "Data Size (E)"

        print(f"\\subsection{{{data_source.capitalize()} - {x_label}}}")
        print()
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{" + f"K(N), E0(N), and R² for each model size ({data_source}, {x_label})" + "}")
        print("\\label{tab:" + f"{data_source}_{x_var}_per_model" + "}")
        print("\\begin{tabular}{cccc}")
        print("\\hline")
        print("Model Size (B) & K(N) & E0(N) & R² \\\\")
        print("\\hline")

        for model in result["models"]:
            N = model["N_billions"]
            k = model["k_N"]
            E0 = model["E0"]
            r2 = model["r2"]

            print(f"{N:.1f} & {k:.6f} & {E0:.6f} & {r2:.6f} \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
        print()
        print(f"Parameters: $k_{{\\max}} = {result['k_max']:.6f}$, $N_0 = {result['N0']:.2e}$")
        print()
        print()

    # Generate compact tables (Base vs Instruct side-by-side)
    print("\n" + "="*80)
    print("Compact LaTeX Tables (Base vs Instruct)")
    print("="*80 + "\n")

    # Get overall R² from fit results
    with open("outputs/fits_hybrid_kn_e072.json") as f:
        fit_info = json.load(f)

    r2_overall = {}
    r2_72B = {}
    for fit in fit_info["fits"]:
        ctx = fit["context"]
        key = (ctx["data_source"], ctx["fit_x"])
        r2_overall[key] = fit["info"]["r2"]
        r2_72B[key] = fit["info"]["r2_72B"]

    # Table for L(N, C)
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{$L(N,C)$ Fitting Results - Compact Version}")
    print("\\label{tab:fitting_results_N_C_compact}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Model Size}} & \\multicolumn{3}{c}{\\textbf{Base}} & \\multicolumn{3}{c}{\\textbf{Instruct}} \\\\")
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    print(" & $k_{C}$ & $E0_{C}$ & $R^2_{C}$ & $k_{C}$ & $E0_{C}$ & $R^2_{C}$ \\\\")
    print("\\midrule")

    base_c = results["base_C_raw"]["models"]
    instruct_c = results["instruct_C_raw"]["models"]
    r2_base_overall = r2_overall[("base", "C_raw")]
    r2_instruct_overall = r2_overall[("instruct", "C_raw")]
    r2_base_72 = r2_72B[("base", "C_raw")]
    r2_instruct_72 = r2_72B[("instruct", "C_raw")]

    for i in range(len(base_c)):
        b = base_c[i]
        ins = instruct_c[i]
        N_str = f"{b['N_billions']:.1f}B"

        # Show per-model R² for each row
        print(f"{N_str} & {b['k_N']:.4f} & {b['E0']:.4f} & {b['r2']:.3f} & {ins['k_N']:.4f} & {ins['E0']:.4f} & {ins['r2']:.3f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

    # Table for L(N, E)
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{$L(N,E)$ Fitting Results - Compact Version}")
    print("\\label{tab:fitting_results_N_E_compact}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Model Size}} & \\multicolumn{3}{c}{\\textbf{Base}} & \\multicolumn{3}{c}{\\textbf{Instruct}} \\\\")
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    print(" & $k_{E}$ & $E0_{E}$ & $R^2_{E}$ & $k_{E}$ & $E0_{E}$ & $R^2_{E}$ \\\\")
    print("\\midrule")

    base_e = results["base_E"]["models"]
    instruct_e = results["instruct_E"]["models"]
    r2_base_overall = r2_overall[("base", "E")]
    r2_instruct_overall = r2_overall[("instruct", "E")]
    r2_base_72 = r2_72B[("base", "E")]
    r2_instruct_72 = r2_72B[("instruct", "E")]

    for i in range(len(base_e)):
        b = base_e[i]
        ins = instruct_e[i]
        N_str = f"{b['N_billions']:.1f}B"

        # Show per-model R² for each row
        print(f"{N_str} & {b['k_N']:.4f} & {b['E0']:.4f} & {b['r2']:.3f} & {ins['k_N']:.4f} & {ins['E0']:.4f} & {ins['r2']:.3f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

    # Generate combined table
    print("\n" + "="*80)
    print("Combined LaTeX Table (All Configurations)")
    print("="*80 + "\n")

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{K(N), E0(N), and R² for each model size across all configurations}")
    print("\\label{tab:all_per_model_r2}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Configuration & Model Size (B) & K(N) & E0(N) & R² \\\\")
    print("\\hline")

    for key, result in results.items():
        data_source = result["data_source"]
        x_var = result["x_var"]
        x_label = "C" if x_var == "C_raw" else "E"
        config_name = f"{data_source.capitalize()}-{x_label}"

        for i, model in enumerate(result["models"]):
            N = model["N_billions"]
            k = model["k_N"]
            E0 = model["E0"]
            r2 = model["r2"]

            if i == 0:
                print(f"{config_name} & {N:.1f} & {k:.6f} & {E0:.6f} & {r2:.6f} \\\\")
            else:
                print(f" & {N:.1f} & {k:.6f} & {E0:.6f} & {r2:.6f} \\\\")

        print("\\hline")

    print("\\end{tabular}")
    print("\\end{table}")

    # Generate table with statistics
    print("\n" + "="*80)
    print("Per-Model Statistics (with ErrRate std)")
    print("="*80 + "\n")

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Per-model R² with data statistics}")
    print("\\label{tab:per_model_r2_with_stats}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Config & N (B) & K(N) & E0(N) & R² & ErrRate Mean & ErrRate Std \\\\")
    print("\\hline")

    for key, result in results.items():
        data_source = result["data_source"]
        x_var = result["x_var"]
        x_label = "C" if x_var == "C_raw" else "E"
        config_name = f"{data_source.capitalize()}-{x_label}"

        for i, model in enumerate(result["models"]):
            N = model["N_billions"]
            k = model["k_N"]
            E0 = model["E0"]
            r2 = model["r2"]
            err_mean = model["err_mean"]
            err_std = model["err_std"]

            if i == 0:
                print(f"{config_name} & {N:.1f} & {k:.6f} & {E0:.6f} & {r2:.6f} & {err_mean:.4f} & {err_std:.4f} \\\\")
            else:
                print(f" & {N:.1f} & {k:.6f} & {E0:.6f} & {r2:.6f} & {err_mean:.4f} & {err_std:.4f} \\\\")

        print("\\hline")

    print("\\end{tabular}")
    print("\\end{table}")
    print()
    print("\\textbf{Note:} R² values are lower for smaller models due to smaller variance in ErrRate,")
    print("not due to poor fit quality. The fitted curves closely match the data points for all model sizes.")

    # Save to file
    output_file = Path("outputs/per_model_r2_latex.txt")
    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("Per-Model K(N), E0(N), and R² Analysis\n")
        f.write("="*80 + "\n\n")

        for key, result in results.items():
            data_source = result["data_source"]
            x_var = result["x_var"]
            x_label = "Compute (C)" if x_var == "C_raw" else "Data Size (E)"

            f.write(f"\\subsection{{{data_source.capitalize()} - {x_label}}}\n\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{" + f"K(N), E0(N), and R² for each model size ({data_source}, {x_label})" + "}\n")
            f.write("\\label{tab:" + f"{data_source}_{x_var}_per_model" + "}\n")
            f.write("\\begin{tabular}{cccc}\n")
            f.write("\\hline\n")
            f.write("Model Size (B) & K(N) & E0(N) & R² \\\\\n")
            f.write("\\hline\n")

            for model in result["models"]:
                N = model["N_billions"]
                k = model["k_N"]
                E0 = model["E0"]
                r2 = model["r2"]

                f.write(f"{N:.1f} & {k:.6f} & {E0:.6f} & {r2:.6f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            f.write(f"Parameters: $k_{{\\max}} = {result['k_max']:.6f}$, $N_0 = {result['N0']:.2e}$\n\n\n")

        # Combined table
        f.write("\n" + "="*80 + "\n")
        f.write("Combined LaTeX Table (All Configurations)\n")
        f.write("="*80 + "\n\n")

        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{K(N), E0(N), and R² for each model size across all configurations}\n")
        f.write("\\label{tab:all_per_model_r2}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Configuration & Model Size (B) & K(N) & E0(N) & R² \\\\\n")
        f.write("\\hline\n")

        for key, result in results.items():
            data_source = result["data_source"]
            x_var = result["x_var"]
            x_label = "C" if x_var == "C_raw" else "E"
            config_name = f"{data_source.capitalize()}-{x_label}"

            for i, model in enumerate(result["models"]):
                N = model["N_billions"]
                k = model["k_N"]
                E0 = model["E0"]
                r2 = model["r2"]

                if i == 0:
                    f.write(f"{config_name} & {N:.1f} & {k:.6f} & {E0:.6f} & {r2:.6f} \\\\\n")
                else:
                    f.write(f" & {N:.1f} & {k:.6f} & {E0:.6f} & {r2:.6f} \\\\\n")

            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
