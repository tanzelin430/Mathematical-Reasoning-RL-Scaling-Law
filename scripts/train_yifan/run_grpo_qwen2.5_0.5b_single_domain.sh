#!/bin/bash
set -x

# =================== Environment Configuration ===================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
export WANDB_API_KEY='1c01f395e45cd03487bdb9c72cbefe7cdef54426'
export DAYTONA_API_KEY="dtn_2d9adc998ab6f2766510546599f5ebd29afb218941bc0e66c02f41f2128021f9"
export STEM_LLM_JUDGE_URL="http://localhost:8040"

# =================== Proxy Configuration ===================
# Ensure proxy settings are preserved
# export https_proxy=https://tanzelin.p:6EEklxJn6slipeJzRoQ4Iy7V4xo58tmUThq8DdnAc1F6rKr0jFXbg9vO92YX@volc-proxy.pjlab.org.cn:13128/
# export http_proxy=https://tanzelin.p:6EEklxJn6slipeJzRoQ4Iy7V4xo58tmUThq8DdnAc1F6rKr0jFXbg9vO92YX@volc-proxy.pjlab.org.cn:13128/
# export WANDB_HTTP_TIMEOUT=300
# =================== Data Configuration ===================
# Use consistent absolute path
# SHARED_DATA_PATH=/home/local/PARTNERS/yz646/Agentic-RL-Scaling-Law/data/guru_verl
SHARED_DATA_PATH=/home/myid/yz44466/Agentic-RL-Scaling-Law/data/guru_verl
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train/
VAL_DATA_DIR=${SHARED_DATA_PATH}/online_eval/
# =================== Output and Checkpoint Configuration ===================
# Save checkpoints and outputs to results directory
# Use absolute path to ensure checkpoints are saved in the correct location
# RESULTS_DIR=/home/local/PARTNERS/yz646/Agentic-RL-Scaling-Law/results
RESULTS_DIR=/home/myid/yz44466/Agentic-RL-Scaling-Law/results
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints
# Create checkpoint directory if it doesn't exist
mkdir -p ${CHECKPOINT_DIR}      
# Choose which domain to train on (uncomment one)
# Option 1: Math domain only
# DOMAIN_NAME="math"
# train_files="['${TRAIN_DATA_DIR}/math__combined_54.4k.parquet']"

# Option 2: Code domain only
# DOMAIN_NAME="code"
# train_files="['${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k.parquet']"

# Option 3: Logic domain only
# DOMAIN_NAME="logic"
# train_files="['${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet']"

# Option 4: STEM domain only
DOMAIN_NAME="stem"
train_files="['${TRAIN_DATA_DIR}/stem__web_3.6k.parquet']"

# Option 5: Math + STEM (original configuration)
# DOMAIN_NAME="math_stem"
# train_files="['${TRAIN_DATA_DIR}/math__combined_54.4k.parquet', '${TRAIN_DATA_DIR}/stem__web_3.6k.parquet']"

# Validation file (optional)
val_files="['${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet']"
# val_files=""  # Uncomment to disable validation

# =================== Model Configuration ===================
MODEL_NAME=Qwen2.5-0.5B-Instruct
BASE_MODEL=Qwen/${MODEL_NAME}

# =================== Logging Configuration ===================
WANDB_PROJECT=agentic_rl_scaling_law
WANDB_EXPERIMENT_NAME=${MODEL_NAME}_${DOMAIN_NAME}_grpo_verl_builtin_check

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
max_prompt_length=1024
max_response_length=8192

# Hardware Platform
num_nodes=1
n_gpus_per_node=8

# Batch sizes (adjusted for GRPO)
train_prompt_bsz=16  # Larger batch for GRPO
n_resp_per_prompt=8  # GRPO needs multiple responses (at least 2, typically 4-8)
train_prompt_mini_bsz=8  # Mini batch size for gradient updates

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
offload=True
gpu_memory_utilization=0.25

# =================== Start GRPO Training ===================
echo "Checkpoints will be saved to: ${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}"

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
    trainer.save_freq=1000 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.resume_mode=disable \
    +reward_model.daytona.api_key="${DAYTONA_API_KEY}" \
    +reward_model.daytona.max_concurrent=8 \
    +reward_model.daytona.timeout=5 \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} $@