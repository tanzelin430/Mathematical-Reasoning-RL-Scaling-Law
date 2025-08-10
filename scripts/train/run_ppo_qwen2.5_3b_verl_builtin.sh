#!/bin/bash
set -x

# =================== Environment Configuration ===================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

# Optional: Add NCCL optimizations if needed
# export NCCL_DEBUG=info
# export NCCL_ALGO=NVLSTree

# =================== Data Mixture ===================
SHARED_DATA_PATH=/root/work/Agentic-RL-Scaling-Law/data/guru_verl
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train/

# Math (54.4k samples)
math_train_path=${TRAIN_DATA_DIR}/math__combined_54.4k.parquet

# Code (1.3k samples)  
codegen_train_path=${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k.parquet

# Logic (111 + 190 samples for training, can use arcagi2 for validation)
logic_arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet
logic_arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet

# STEM (3.6k samples)
stem_train_path=${TRAIN_DATA_DIR}/stem__web_3.6k.parquet

# Configure training files - all 4 domains
train_files="['${math_train_path}', '${codegen_train_path}', '${logic_arcagi1_train_path}', '${stem_train_path}']"

# Use logic arcagi2 as validation (optional, can be set to null)
val_files="['${logic_arcagi2_train_path}']"
# val_files="null"  # Uncomment to disable validation

# =================== Model Configuration ===================
BASE_MODEL=/fs-computility/mabasic/shared/models/Qwen2.5-3B

# =================== Logging Configuration ===================
WANDB_PROJECT=agentic_rl_scaling_law
WANDB_EXPERIMENT_NAME=qwen2.5_3b_verl_builtin_all_domains

# =================== Output and Checkpoint Configuration ===================
# Save checkpoints and outputs to results directory
RESULTS_DIR=/fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/results
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints

# =================== RL Training Parameters ===================
# Algorithm settings (using GAE like your original)
adv_estimator=gae  # or grpo for better performance

# KL penalty settings
use_kl_in_reward=True
kl_coef=0.02
use_kl_loss=False
kl_loss_coef=0.0

# PPO clipping
clip_ratio_low=0.2
clip_ratio_high=0.2

# Sequence length limits
max_prompt_length=2048
max_response_length=2048

# Batch sizes (adjusted for 2x A800 GPUs)
train_prompt_bsz=8  # Total batch size for training
gen_prompt_bsz=$((train_prompt_bsz * 1))  # Generation batch size
n_resp_per_prompt=4  # Number of responses per prompt (reduced from 16 for memory)
train_prompt_mini_bsz=4  # Mini batch size for gradient updates

# Sampling parameters
temperature=1.0
top_p=1.0
top_k=-1  # -1 for vLLM rollout

# Model parallelism settings (for 2 GPUs)
gen_tp=2  # Tensor parallel size for generation
sp_size=1  # Sequence parallel size (1 for small setup)

# Memory optimization
offload=False  # Set to True if OOM issues
gpu_memory_utilization=0.5  # Increased from 0.4 for better utilization

# =================== Start RL Training ===================
# Using verl.trainer.main_ppo directly (no custom reward function needed)
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
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
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
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=${offload} \
    critic.model.fsdp_config.optimizer_offload=${offload} \
    trainer.logger='["console"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=3 \
    trainer.critic_warmup=5 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} $@