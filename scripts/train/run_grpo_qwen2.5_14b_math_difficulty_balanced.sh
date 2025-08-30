#!/bin/bash
set -x

# =================== Environment Configuration ===================
# GPU Configuration
TRAINING_GPUS="0,1,2,3,4,5,6,7"  # GPUs for training
AUTHOR_NAME="tanzl"
export WANDB_DIR=~/Agentic-RL-Scaling-Law/wandb_${AUTHOR_NAME}
mkdir -p $WANDB_DIR
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
export WANDB_API_KEY='1328200322cf91b211452e5b4ca7ce9148bc7250'
export WANDB_MODE=offline

# =================== Data Configuration ===================
# Use our difficulty-balanced math data
DIFFICULTY_DATA_FILE="/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/data/difficulty_balanced_math/math__difficulty_balanced_26804.parquet"
train_files="['${DIFFICULTY_DATA_FILE}']"

# Validation files - just math for this focused training
VAL_DATA_DIR=../../data/guru_verl/online_eval/
val_files="['${VAL_DATA_DIR}/math__math_500.parquet']"

# =================== Output and Checkpoint Configuration ===================
RESULTS_DIR=/mnt/shared-storage-user/ma4agi-gpu/RLscaleckpt
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints
mkdir -p ${CHECKPOINT_DIR}      

# =================== Model Configuration ===================
MODEL_NAME=Qwen2.5-14B
BASE_MODEL=/mnt/shared-storage-user/ma4agi-gpu/data/model/${MODEL_NAME}

# =================== GRPO Training Parameters ===================
# Algorithm settings - GRPO specific
adv_estimator=grpo  # GRPO estimator

# KL settings for GRPO
use_kl_in_reward=False  # GRPO doesn't use KL in reward
use_kl_loss=True  # GRPO uses KL loss instead
kl_loss_coef=0.001  # Standard GRPO KL coefficient
kl_loss_type=low_var_kl  # Low variance KL for GRPO

# PPO clipping (still used in GRPO)
clip_ratio_low=0.2
clip_ratio_high=0.2

# Sequence length limits
max_prompt_length=2048
max_response_length=4096

# Hardware Platform
num_nodes=1
n_gpus_per_node=8

# Batch sizes (adjusted for GRPO and 14B model)
train_prompt_bsz=512  # Batch size
n_resp_per_prompt=8  # GRPO needs multiple responses
train_prompt_mini_bsz=128  # Mini batch size

# Calculate epochs based on dataset size (26804 samples)
DATA_SIZE=26804
Required_Train_step=1000
EPOCHS=$((Required_Train_step * train_prompt_bsz / DATA_SIZE))

# =================== Logging Configuration ===================
WANDB_PROJECT=agentic_rl_scaling_law
WANDB_EXPERIMENT_NAME=${MODEL_NAME}_math_difficulty_balanced_grpo_verl_builtin_${EPOCHS}

# Dynamic batch size configuration
use_dynamic_bsz=True

# Calculate max sequence length from input/output settings
max_seq_length=$((max_prompt_length + max_response_length))  # 6144

# Token limit multipliers based on VeRL official example
actor_seq_multiplier=8  # Actor should be ~2x max sequence length
rollout_seq_multiplier=10

# Calculate token limits for GRPO (no critic needed)
actor_ppo_max_token_len=$((max_seq_length * actor_seq_multiplier))
rollout_log_prob_max_token_len=$((max_seq_length * rollout_seq_multiplier))

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

export CUDA_VISIBLE_DEVICES=${TRAINING_GPUS}

# =================== Start GRPO Training ===================
echo "Training with difficulty-balanced math data from: ${DIFFICULTY_DATA_FILE}"
echo "Dataset contains 26,804 samples sorted from easiest to hardest"
echo "Checkpoints will be saved to: ${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}/hydra_outputs \
    hydra.sweep.dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}/hydra_multirun \
    hydra.job.chdir=False \
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
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${num_nodes} \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=${EPOCHS} \
    trainer.resume_mode=auto \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} $@