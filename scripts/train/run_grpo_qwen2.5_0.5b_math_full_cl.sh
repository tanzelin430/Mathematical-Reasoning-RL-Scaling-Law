#!/bin/bash
set -x

# =================== Environment Configuration ===================
# GPU Configuration
TRAINING_GPUS="0,1,2,3"
AUTHOR_NAME="tanzl"
export WANDB_DIR=~/Agentic-RL-Scaling-Law/wandb
mkdir -p $WANDB_DIR
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
export STEM_LLM_JUDGE_URL="http://localhost:${STEM_JUDGE_PORT}"
export WANDB_MODE=offline

# =================== Data Configuration ===================
OUTPUT_DATA_DIR="../../data/math_curriculum"

if [ $? -ne 0 ]; then
    echo "ERROR: Data preparation failed!"
    exit 1
fi

# Use the generated files
TRAIN_FILE="${OUTPUT_DATA_DIR}/math__full_difficulty_ordered_train_53904.parquet"
TEST_FILE="${OUTPUT_DATA_DIR}/math__full_difficulty_ordered_test_500.parquet"

# Original validation data directory
SHARED_DATA_PATH=../../data/guru_verl
VAL_DATA_DIR=${SHARED_DATA_PATH}/online_eval/

# Training data - single file format
train_files="['${TRAIN_FILE}']"

# Validation files - include our test split + original eval files + AIME + GSM8K
val_files="['${TEST_FILE}', '${VAL_DATA_DIR}/math__math_500.parquet', '${VAL_DATA_DIR}/logic__zebra_puzzle_dataset_200.parquet', '${VAL_DATA_DIR}/stem__supergpqa_200.parquet', '${VAL_DATA_DIR}/codegen__humaneval_164.parquet', '${VAL_DATA_DIR}/aime2024.parquet', '${VAL_DATA_DIR}/gsm8k.parquet', '${VAL_DATA_DIR}/amc2023.parquet']"

# Batch sizes
train_prompt_bsz=512
n_resp_per_prompt=8
train_prompt_mini_bsz=128

# =================== Output and Checkpoint Configuration ===================
RESULTS_DIR=/mnt/shared-storage-user/ma4agi-gpu/RLscaleckpt
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints
mkdir -p ${CHECKPOINT_DIR}      

# =================== Model Configuration ===================
MODEL_NAME=Qwen2.5-0.5B
BASE_MODEL=/mnt/shared-storage-user/ma4agi-gpu/data/model/${MODEL_NAME}

# =================== Logging Configuration ===================
WANDB_PROJECT=agentic_rl_scaling_law

DOMAIN_NAME="math_curriculum_full"
WANDB_EXPERIMENT_NAME=${MODEL_NAME}_${DOMAIN_NAME}_grpo_verl_builtin_CL_run1

# =================== GRPO Training Parameters ===================
adv_estimator=grpo

# KL settings
use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# PPO clipping
clip_ratio_low=0.2
clip_ratio_high=0.2

# Sequence length limits
max_prompt_length=2048
max_response_length=4096

# Hardware Platform
num_nodes=1
n_gpus_per_node=4

EPOCHS=1

use_dynamic_bsz=True

max_seq_length=$((max_prompt_length + max_response_length))

actor_seq_multiplier=12
rollout_seq_multiplier=14
actor_ppo_max_token_len=$((max_seq_length * actor_seq_multiplier))
rollout_log_prob_max_token_len=$((max_seq_length * rollout_seq_multiplier))

# Sampling parameters
temperature=1.0
top_p=1.0
top_k=-1

val_temperature=0.7
val_top_p=1.0
val_top_k=-1

gen_tp=1
sp_size=1

offload=False
gpu_memory_utilization=0.65

export CUDA_VISIBLE_DEVICES=${TRAINING_GPUS}

# =================== Start GRPO Training ===================
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
    data.shuffle=False \
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
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_seqs=64 \
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
    trainer.test_freq=1 \
    trainer.save_freq=10 \
    trainer.total_epochs=${EPOCHS} \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} $@