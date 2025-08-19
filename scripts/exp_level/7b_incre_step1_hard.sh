#!/bin/bash
set -x

# =================== Environment Configuration ===================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

# =================== Data Configuration ===================
SHARED_DATA_PATH=/fs-computility/mabasic/xuexiangyuan/workspace/agentic-rl-scaling-law/data/guru_verl_level
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train

# =================== Output and Checkpoint Configuration ===================
# Save checkpoints and outputs to results directory
# Use absolute path to ensure checkpoints are saved in the correct location
RESULTS_DIR=/fs-computility/mabasic/xuexiangyuan/workspace/agentic-rl-scaling-law/results
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints
# Create checkpoint directory if it doesn't exist
mkdir -p ${CHECKPOINT_DIR}

# Difficulty-based data splits (processed by pre_verl_level.py)
easy_train_path=${TRAIN_DATA_DIR}/combined_easy.parquet
medium_train_path=${TRAIN_DATA_DIR}/combined_medium.parquet
hard_train_path=${TRAIN_DATA_DIR}/combined_hard.parquet

# Configure training files - all 3 difficulty levels
train_files="['${hard_train_path}']"

# Use medium difficulty as validation (optional, can be set to null)
val_files="['${medium_train_path}']"
# val_files="null"  # Uncomment to disable validation

# =================== Model Configuration ===================
BASE_MODEL=/PATH/TO/LAST/CHECKPOINT/

# =================== Logging Configuration ===================
WANDB_PROJECT=agentic_rl_scaling_law
WANDB_EXPERIMENT_NAME=exp_level_7b_incre_step1_hard

# =================== GRPO Training Parameters ===================
# Algorithm settings - GRPO specific
adv_estimator=grpo  # Changed from gae to grpo

# KL settings for GRPO
use_kl_in_reward=False  # GRPO doesn't use KL in reward
use_kl_loss=True  # GRPO uses KL loss instead
kl_loss_coef=0.001  # Standard GRPO KL coefficient
kl_loss_type=low_var_kl  # Low variance KL for GRPO

# PPO clipping (still used in GRPO)
clip_ratio_low=0.2
clip_ratio_high=0.2

# Sequence length limits
max_prompt_length=4096
max_response_length=4096

# Hardware Platform
num_nodes=1
n_gpus_per_node=8

# Batch sizes (adjusted for GRPO)
train_prompt_bsz=256  # Larger batch for GRPO
n_resp_per_prompt=8  # GRPO needs multiple responses (at least 2, typically 4-8)
train_prompt_mini_bsz=128  # Mini batch size for gradient updates

# Dynamic batch size configuration
use_dynamic_bsz=True

# Calculate max sequence length from input/output settings
max_seq_length=$((max_prompt_length + max_response_length))  # 1024 + 8192 = 9216

# Token limit multipliers based on VeRL official example
# For GRPO, we don't need critic, so only actor and rollout
actor_seq_multiplier=2  # Actor should be ~3x max sequence length

# Calculate token limits for GRPO (no critic needed)
actor_ppo_max_token_len=$((max_seq_length * actor_seq_multiplier))  # 27648
rollout_log_prob_max_token_len=${actor_ppo_max_token_len}  # Same as actor

# Sampling parameters
temperature=1.0
top_p=1.0
top_k=-1

# Model parallelism settings
gen_tp=2
sp_size=1

# Memory optimization
offload=False
gpu_memory_utilization=0.65

# =================== Start RL Training ===================
# Using verl.trainer.main_ppo directly (no custom reward function needed)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.trust_remote_code=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${rollout_log_prob_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${num_nodes} \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} $@
