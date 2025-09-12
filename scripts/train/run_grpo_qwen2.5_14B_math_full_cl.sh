#!/bin/bash
set -x

# =================== Environment Configuration ===================
# STEM LLM Judge Configuration
# STEM_JUDGE_PORT=8040  # Port for STEM LLM Judge server
# STEM_JUDGE_MODEL_PATH="/mnt/shared-storage-user/ma4agi-gpu/data/model/general-verifier/"  # Path to STEM Judge model
# STEM_JUDGE_GPU=0,1,2,3  # GPU for STEM LLM Judge (separate from training GPUs)

# GPU Configuration
TRAINING_GPUS="0,1,2,3,4,5,6,7"  # GPUs for training (excluding GPU 0 which is used by vLLM)
AUTHOR_NAME="tanzl"
export WANDB_DIR=/home/yinzhenfei/Agentic-RL-Scaling-Law/wandb_tanzl/wandb
mkdir -p $WANDB_DIR
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
export WANDB_API_KEY='1328200322cf91b211452e5b4ca7ce9148bc7250'
# export DAYTONA_API_KEY="dtn_2d9adc998ab6f2766510546599f5ebd29afb218941bc0e66c02f41f2128021f9"
export STEM_LLM_JUDGE_URL="http://localhost:${STEM_JUDGE_PORT}"
export WANDB_MODE=offline

# =================== SandboxFusion Configuration ===================
# Enable SandboxFusion for code execution
# export CODER1_EXEC=sandboxfusion
# NUM_SANDBOX=8
# BASE_PORT=10086
# SERVERS=""
# for i in $(seq 0 $((NUM_SANDBOX-1))); do
#     PORT=$((BASE_PORT + i))
#     [ -z "$SERVERS" ] && SERVERS="localhost:$PORT" || SERVERS="$SERVERS,localhost:$PORT"
# done
# export SANDBOX_FUSION_SERVERS="$SERVERS"
# Optional: For multiple servers (load balancing)
# export SANDBOX_FUSION_SERVERS="server1:8080,server2:8080,server3:8080"

# =================== Data Configuration ===================
# Prepare difficulty-ordered math dataset
OUTPUT_DATA_DIR="../../data/math_curriculum"
# mkdir -p ${OUTPUT_DATA_DIR}

# echo "Preparing curriculum math dataset..."
# python3 /home/tanzelin-p/Agentic-RL-Scaling-Law/src/data/prepare_math_by_difficulty_full.py \
#     --input_file="/mnt/shared-storage-user/ma4agi-gpu/data/dataset/guru-RL-92k/train/math__combined_54.4k.parquet" \
#     --output_dir="${OUTPUT_DATA_DIR}" \
#     --test_size=500

# Check if data preparation was successful
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

# Batch sizes (adjusted for GRPO and 7B model)
train_prompt_bsz=512  # Reduced from 256 for mixed domain
n_resp_per_prompt=8  # GRPO needs multiple responses
train_prompt_mini_bsz=128  # Reduced for mixed domain

# =================== Output and Checkpoint Configuration ===================
# Save checkpoints and outputs to results directory
# Use absolute path to ensure checkpoints are saved in the correct location
RESULTS_DIR=/mnt/shared-storage-user/ma4agi-gpu/RLscaleckpt
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints
# Create checkpoint directory if it doesn't exist
mkdir -p ${CHECKPOINT_DIR}      

# =================== Model Configuration ===================
MODEL_NAME=Qwen2.5-14B-Instruct
BASE_MODEL=/mnt/shared-storage-user/ma4agi-gpu/data/model/${MODEL_NAME}

# =================== Logging Configuration ===================
WANDB_PROJECT=agentic_rl_scaling_law

DOMAIN_NAME="math_curriculum_full"
WANDB_EXPERIMENT_NAME=${MODEL_NAME}_${DOMAIN_NAME}_grpo_verl_builtin_CL_run0

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
n_gpus_per_node=8  # Default to 8 GPUs

# Set epochs to 1 as requested
EPOCHS=1

# Dynamic batch size configuration
use_dynamic_bsz=True

# Calculate max sequence length from input/output settings
max_seq_length=$((max_prompt_length + max_response_length))  # 1024 + 8192 = 9216

# Token limit multipliers based on VeRL official example
# For GRPO, we don't need critic, so only actor and rollout
actor_seq_multiplier=8  # Actor should be ~2x max sequence length
rollout_seq_multiplier=10
# Calculate token limits for GRPO (no critic needed)
actor_ppo_max_token_len=$((max_seq_length * actor_seq_multiplier))  # 18432
rollout_log_prob_max_token_len=$((max_seq_length * rollout_seq_multiplier))  # Same as actor

# Sampling parameters
temperature=1.0
top_p=1.0
top_k=-1

#validation parameters
val_temperature=0.7
val_top_p=1.0
val_top_k=-1

# Model parallelism settings
gen_tp=1
sp_size=1

# Memory optimization
offload=False
gpu_memory_utilization=0.65  # Reduced from 0.65 for stability with mixed domains

# =================== Conditional STEM LLM Judge Setup ===================
start_stem_judge() {
    echo "Starting STEM LLM Judge server for mixed domain training..."
    echo "Using GPU ${STEM_JUDGE_GPU} for vLLM server..."
    
    # Start vLLM server in background with dedicated GPU
    CUDA_VISIBLE_DEVICES=${STEM_JUDGE_GPU} python -m vllm.entrypoints.openai.api_server \
        --host localhost \
        --port ${STEM_JUDGE_PORT} \
        --model ${STEM_JUDGE_MODEL_PATH} \
        --served-model-name TIGER-Lab/general-verifier \
        --gpu-memory-utilization 0.1 \
        --max-model-len 1024 \
        --tensor-parallel-size 4 \
        --trust-remote-code > stem_judge.log 2>&1 &
    
    STEM_JUDGE_PID=$!
    echo "STEM Judge server started with PID: $STEM_JUDGE_PID"
    
    # Wait for server to be ready
    echo "Waiting for STEM Judge server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:${STEM_JUDGE_PORT}/health > /dev/null 2>&1; then
            echo "✅ STEM Judge server is ready!"
            return 0
        fi
        echo -n "."
        sleep 3
    done
    
    echo "❌ STEM Judge server failed to start. Check stem_judge.log"
    exit 1
}

# Cleanup function
cleanup_stem_judge() {
    if [ ! -z "$STEM_JUDGE_PID" ]; then
        echo "Stopping STEM Judge server (PID: $STEM_JUDGE_PID)..."
        kill $STEM_JUDGE_PID 2>/dev/null
        wait $STEM_JUDGE_PID 2>/dev/null
        echo "STEM Judge server stopped."
    fi
}

# Set trap to cleanup on script exit
# trap cleanup_stem_judge EXIT

# Since we're training on mixed domains including STEM, start the STEM judge
# echo "Mixed domain training includes STEM. Starting STEM LLM Judge..."
# start_stem_judge
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
    # +reward_model.daytona.api_key="${DAYTONA_API_KEY}" \
    # +reward_model.daytona.max_concurrent=8 \
    # +reward_model.daytona.timeout=5 \
    # actor_rollout_ref.rollout.enable_chunked_prefill=True \
    # actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    # actor_rollout_ref.rollout.max_model_len=10240 \