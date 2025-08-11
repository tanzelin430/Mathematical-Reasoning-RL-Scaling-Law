#!/bin/bash
set -x

# =================== Environment Configuration ===================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

# =================== Data Configuration ===================
# Use consistent absolute path
SHARED_DATA_PATH=/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/data/guru_verl
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train/

# =================== Output and Checkpoint Configuration ===================
# Save checkpoints and outputs to results directory
# Use absolute path to ensure checkpoints are saved in the correct location
RESULTS_DIR=/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/results
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints
# Create checkpoint directory if it doesn't exist
mkdir -p ${CHECKPOINT_DIR}      
# Choose which domain to train on (uncomment one)
# Option 1: Math domain only
DOMAIN_NAME="math"
train_files="['${TRAIN_DATA_DIR}/math__combined_54.4k.parquet']"

# Option 2: Code domain only
# DOMAIN_NAME="code"
# train_files="['${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k.parquet']"

# Option 3: Logic domain only
# DOMAIN_NAME="logic"
# train_files="['${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet']"

# Option 4: STEM domain only
# DOMAIN_NAME="stem"
# train_files="['${TRAIN_DATA_DIR}/stem__web_3.6k.parquet']"

# Option 5: Math + STEM (original configuration)
# DOMAIN_NAME="math_stem"
# train_files="['${TRAIN_DATA_DIR}/math__combined_54.4k.parquet', '${TRAIN_DATA_DIR}/stem__web_3.6k.parquet']"

# Validation file (optional)
val_files="['${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet']"
# val_files="null"  # Uncomment to disable validation

# =================== Model Configuration ===================
MODEL_NAME=Qwen2.5-7B-Instruct
BASE_MODEL=/fs-computility/mabasic/shared/models/${MODEL_NAME}

# =================== Logging Configuration ===================
WANDB_PROJECT=agentic_rl_scaling_law
WANDB_EXPERIMENT_NAME=qwen2.5_7b_Instruct_${DOMAIN_NAME}_verl_builtin

# =================== RL Training Parameters ===================
# Algorithm settings
adv_estimator=gae  # or grpo

# KL penalty settings
use_kl_in_reward=False
kl_coef=0.02
use_kl_loss=False
kl_loss_coef=0.0

# PPO clipping
clip_ratio_low=0.2
clip_ratio_high=0.2

# Sequence length limits
max_prompt_length=2048
max_response_length=8192

#Hardware Platform
num_nodes=1
n_gpus_per_node=8

# Batch sizes (adjusted for 2x A800 GPUs)
train_prompt_bsz=256  # Smaller batch for single domain
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=2  # Number of responses per prompt
train_prompt_mini_bsz=128  # Mini batch size for gradient updates
# micro_batch_size_per_gpu=8  # Deprecated when using dynamic batch size

# Dynamic batch size configuration
use_dynamic_bsz=True

# Global maximum tokens per GPU setting
# For 8x A800 GPUs with Qwen2.5-7B model, we can use a more aggressive setting
# This controls the maximum tokens per micro-batch on each GPU
max_token_per_gpu=12288  
# Alternative settings:
# max_token_per_gpu=16384  # 16K tokens - balanced setting
# max_token_per_gpu=32768  # 32K tokens - aggressive (may cause OOM)

# Use the global setting for all components
actor_ppo_max_token_len=${max_token_per_gpu}
infer_ppo_max_token_len=${max_token_per_gpu}

# Sampling parameters
temperature=1.0
top_p=1.0
top_k=-1


# Model parallelism settings
gen_tp=2
sp_size=1

# Memory optimization
offload=False
gpu_memory_utilization=0.3



# =================== Start RL Training ===================
echo "Checkpoints will be saved to: ${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.type=adaptive \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.kl_ctrl.target_kl=0.01 \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.trust_remote_code=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
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
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
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
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
    critic.optim.lr=1e-5 \
    critic.model.path=${BASE_MODEL} \
    critic.model.trust_remote_code=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    critic.model.fsdp_config.param_offload=${offload} \
    critic.model.fsdp_config.optimizer_offload=${offload} \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${num_nodes} \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.critic_warmup=0 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} $@