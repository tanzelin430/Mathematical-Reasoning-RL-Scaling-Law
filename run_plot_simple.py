#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

import data_proc
import plot_data
import config
import matplotlib.pyplot as plt

def main():
    model_type = "instruct"

    # df = data_proc.load_and_preprocess(config.CSV_INSTRUCT_RUNS if model_type == "instruct" else config.CSV_BASE_RUNS)
    df = data_proc.load_and_preprocess([config.CSV_INSTRUCT_RUNS[0]])
    plot_data.plot_curves_with_smooth(
        df,
        curve_column='N',
        x_column='E',
        y_column='T',
        use_scatter=True,
    )
    plt.show()

if __name__ == "__main__":
    main()