#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Use VeRL formatted math data
train_files='["/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/data/guru_verl/train/math__combined_54.4k.parquet"]'
val_files='["/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/data/guru_verl/train/math__combined_54.4k.parquet"]'

python3 -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs \
    --config-name=ppo_1.5b_minimal \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    trainer.val_before_train=False