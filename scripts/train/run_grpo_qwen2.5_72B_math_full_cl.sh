#!/bin/bash
set -x

# =================== Environment Configuration ===================
# GPU Configuration
# TRAINING_GPUS="0,1,2,3,4,5,6,7"
AUTHOR_NAME="tanzl"
export WANDB_DIR=/mnt/shared-storage-user/ma4agi-gpu/wandb_tanzl/wandb
mkdir -p $WANDB_DIR
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
# export NCCL_IB_GID_INDEX=3
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
export NCCL_DEBUG=INFO
# export STEM_LLM_JUDGE_URL="http://localhost:${STEM_JUDGE_PORT}"
export WANDB_MODE=offline

OUTPUT_DATA_DIR="/mnt/shared-storage-user/ma4agi-gpu/data/dataset/tzl_data/math_curriculum"

if [ $? -ne 0 ]; then
    echo "ERROR: Data preparation failed!"
    exit 1
fi

# Use the generated files
TRAIN_FILE="${OUTPUT_DATA_DIR}/math__full_difficulty_ordered_train_53904.parquet"
TEST_FILE="${OUTPUT_DATA_DIR}/math__full_difficulty_ordered_test_500.parquet"

# Original validation data directory
SHARED_DATA_PATH=/mnt/shared-storage-user/ma4agi-gpu/data/dataset/tzl_data/guru_verl
VAL_DATA_DIR=${SHARED_DATA_PATH}/online_eval/

# Training data - single file format
train_files="['${TRAIN_FILE}']"

val_files="['${TEST_FILE}', '${VAL_DATA_DIR}/math__math_500.parquet', '${VAL_DATA_DIR}/logic__zebra_puzzle_dataset_200.parquet', '${VAL_DATA_DIR}/stem__supergpqa_200.parquet', '${VAL_DATA_DIR}/codegen__humaneval_164.parquet', '${VAL_DATA_DIR}/aime2024.parquet', '${VAL_DATA_DIR}/gsm8k.parquet', '${VAL_DATA_DIR}/amc2023.parquet']"

train_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=128

# =================== Output and Checkpoint Configuration ===================
RESULTS_DIR=/mnt/shared-storage-user/ma4agi-gpu/RLscaleckpt
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints
mkdir -p ${CHECKPOINT_DIR}      

# =================== Model Configuration ===================
MODEL_NAME=Qwen2.5-72B
BASE_MODEL=/mnt/shared-storage-user/ma4agi-gpu/data/model/${MODEL_NAME}

# =================== Logging Configuration ===================
WANDB_PROJECT=agentic_rl_scaling_law

DOMAIN_NAME="math_curriculum_full"
WANDB_EXPERIMENT_NAME=${MODEL_NAME}_${DOMAIN_NAME}_grpo_verl_builtin_CL_testrun0

# =================== GRPO Training Parameters ===================
adv_estimator=grpo

use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.2



EPOCHS=1




# Sequence length limits
max_prompt_length=1024
max_response_length=2048

# Hardware Platform
num_nodes=4
n_gpus_per_node=8

max_seq_length=$((max_prompt_length + max_response_length)) 

use_dynamic_bsz=False


# Sampling parameters
temperature=1.0

val_temperature=0.7

gen_tp=8
gen_pp=4
offload=False
gpu_memory_utilization=0.7  # For vLLM

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
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    +actor_rollout_ref.rollout.pipeline_model_parallel_size=${gen_pp} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
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
    trainer.nnodes=${num_nodes} \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} $@
