#!/bin/bash

# Single Node RL Training Script (No SLURM)
# This is a simplified version of the multinode script for single-node environments

# =================== Environment Check ===================
echo "Starting single-node RL training setup..."

# Check if we're in a SLURM environment
if command -v srun &> /dev/null; then
    echo "SLURM detected, but this script is for non-SLURM environments."
    echo "Please use example_multinode_rl_qwen32b_base.sh instead."
    exit 1
fi

# =================== Basic Configuration ===================
export WANDB_API_KEY="a8a9253a12a110b9d7609bb8dcec38ece39158f0"
export WANDB_MODE=online
export WANDB_SAVE_CODE=true
export WANDB_ENTITY="aoge"  # Set wandb account/entity
export CUDA_LAUNCH_BLOCKING=1
# Get system information
HOSTNAME=$(hostname)
HOST_IP=$(hostname -I | awk '{print $1}')
CPUS=$(nproc)
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)

echo "System Information:"
echo "  Hostname: $HOSTNAME"
echo "  IP: $HOST_IP"
echo "  CPUs: $CPUS"
echo "  GPUs: $GPU_COUNT"

# Set environment variables
export head_node=$HOSTNAME
export head_node_ip=$HOST_IP
export worker_num=1
export SLURM_NNODES=1
export SLURM_JOB_ID=$$
export SLURM_JOB_NAME="logic-Qwen2.5-7B-Instruct"
export SLURM_CPUS_PER_TASK=$CPUS

# Network settings
export NCCL_DEBUG=info
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

# Fix GLIBC compatibility by prioritizing conda environment libraries
export LD_LIBRARY_PATH="/root/miniconda3/envs/Reasoning360/lib:$LD_LIBRARY_PATH"

# Ray settings
port=6379
address_head=$head_node_ip:$port

# =================== Data Configuration ===================
SHARED_DATA_PATH=/fs-computility/mabasic/shared/models/0807/guru-RL-92k
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train/
TEST_DATA_DIR=${SHARED_DATA_PATH}/offline_eval/

# Math data path lists
math_train_path_list="['${TRAIN_DATA_DIR}/math__combined_54.4k.parquet']"
math_test_path_list="['${TEST_DATA_DIR}/math__math_500.parquet', '${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet']"

# Code data path lists
code_train_path_list="['${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k.parquet', '${TRAIN_DATA_DIR}/codegen__livecodebench_440.parquet', '${TRAIN_DATA_DIR}/codegen__primeintellect_7.5k.parquet', '${TRAIN_DATA_DIR}/codegen__taco_8.8k.parquet']"
code_test_path_list="['${TEST_DATA_DIR}/codegen__humaneval_164.parquet', '${TEST_DATA_DIR}/codegen__livecodebench_279.parquet', '${TEST_DATA_DIR}/codegen__mbpp_500.parquet']"

# Logic data path lists
logic_train_path_list="['${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet', '${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet', '${TRAIN_DATA_DIR}/logic__barc_1.6k.parquet', '${TRAIN_DATA_DIR}/logic__graph_logical_1.2k.parquet', '${TRAIN_DATA_DIR}/logic__ordering_puzzle_1.9k.parquet', '${TRAIN_DATA_DIR}/logic__zebra_puzzle_1.3k.parquet']"
logic_test_path_list="['${TEST_DATA_DIR}/logic__arcagi1_400.parquet', '${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_200.parquet']"

# Combined train and test files
train_files="[${logic_train_path_list:1:-1}]"
test_files="[${logic_test_path_list:1:-1}]"
echo "Train files: $train_files"
echo "Test files: $test_files"

# =================== Model Configuration ===================
BASE_MODEL=/fs-computility/mabasic/shared/models/Qwen2.5-7B-Instruct

# =================== Logging ===================
WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=${SLURM_JOB_ID}-${SLURM_JOB_NAME}-${BASE_MODEL##*/}

# =================== Ray Cluster Setup ===================
echo "Setting up Ray cluster..."

# Stop existing Ray
ray stop 2>/dev/null || true
sleep 5

# Clean up
rm -rf /tmp/ray/ray_current_cluster 2>/dev/null || true

# Start Ray head node
echo "Starting Ray head node with $GPU_COUNT GPUs and $CPUS CPUs"
ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "$CPUS" --num-gpus "$GPU_COUNT" --include-dashboard=True

sleep 5

# Verify Ray is running
if ! ray status &>/dev/null; then
    echo "ERROR: Ray failed to start properly"
    exit 1
fi

echo "Ray cluster started successfully!"

# =================== Training Configuration ===================
# RL Config (based on example_multinode_rl_qwen32b_base.sh)
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

# Reduced batch sizes for single node
train_prompt_bsz=256  # Reduced from 512 for single node
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=8  # Reduced from 16 for single node
train_prompt_mini_bsz=16  # Reduced from 64 for single node

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1  # -1 for vLLM rollout

# Single node specific settings
sp_size=1  # Reduced from 8 for single node
gen_tp=1  # Reduced from 4 for single node
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))
offload=False  # Disabled for single node to improve performance
# vLLM maximum model length aligned to prompt+response to avoid runtime errors
vllm_max_model_len=$((max_prompt_length + max_response_length))

# =================== Start Training ===================
echo "Starting RL training..."

python -m verl.recipe.dapo.src.main_dapo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_model_len=${vllm_max_model_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.rollout.multi_turn.enable=False \
    +actor_rollout_ref.rollout.mode="sync" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=async_dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    +trainer.wandb_entity="hengao" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$GPU_COUNT \
    trainer.nnodes=1 \
    trainer.save_freq=400000 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    +trainer.val_generations_to_log_to_wandb=30 \
    trainer.resume_mode=auto

echo "Training completed!"
