#!/bin/bash
# Example script to plot WandB offline data

# Make scripts executable
chmod +x plot_verl_wandb_offline.py
chmod +x wandb_offline_visualizer.py

echo "===== Simple VeRL WandB Plotting Example ====="

# Example 1: Plot the latest run
echo "1. Plotting latest run..."
python plot_verl_wandb_offline.py \
    --wandb-dir ~/Agentic-RL-Scaling-Law/wandb_tanzl \
    --latest-only \
    --output-dir ./plots/latest_run

# Example 2: Plot all runs matching a pattern
echo -e "\n2. Plotting all Qwen2.5-14B runs..."
python plot_verl_wandb_offline.py \
    --wandb-dir ~/Agentic-RL-Scaling-Law/wandb_tanzl \
    --run-pattern "Qwen2.5-14B" \
    --output-dir ./plots/qwen14b_runs

# Example 3: Plot specific experiment
echo -e "\n3. Plotting specific experiment..."
python plot_verl_wandb_offline.py \
    --wandb-dir ~/Agentic-RL-Scaling-Law/wandb_tanzl \
    --run-pattern "grpo_verl_builtin" \
    --output-dir ./plots/grpo_experiment

echo -e "\n===== Advanced WandB Visualizer Example ====="

# Example 4: Use the advanced visualizer for multiple directories
echo "4. Using advanced visualizer..."
python wandb_offline_visualizer.py \
    --wandb-dirs ~/Agentic-RL-Scaling-Law/wandb_tanzl \
    --find-runs \
    --output-dir ./plots/all_runs \
    --smooth-factor 0.9

echo -e "\nAll plots have been generated! Check the output directories for results."