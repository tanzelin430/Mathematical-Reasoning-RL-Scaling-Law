#!/bin/bash
set -x

# Configure CUDA for optimal performance
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 四个领域的训练数据文件
# Math: 54,404 samples
# Logic: 111 samples  
# Code: 1,272 samples
# STEM: 3,591 samples
# Total: ~59,378 samples
TRAIN_FILES="['/root/work/Agentic-RL-Scaling-Law/data/guru_verl/train/math__combined_54.4k.parquet', '/root/work/Agentic-RL-Scaling-Law/data/guru_verl/train/logic__arcagi1_111.parquet', '/root/work/Agentic-RL-Scaling-Law/data/guru_verl/train/codegen__leetcode2k_1.3k.parquet', '/root/work/Agentic-RL-Scaling-Law/data/guru_verl/train/stem__web_3.6k.parquet']"

# 使用logic的另一个文件作为验证集（如果有的话）
VAL_FILES="['/root/work/Agentic-RL-Scaling-Law/data/guru_verl/train/logic__arcagi2_190.parquet']"

# Model path
MODEL_PATH="/fs-computility/mabasic/shared/models/Qwen2.5-3B"

# Reward function path
REWARD_FN_PATH="/root/work/Agentic-RL-Scaling-Law/src/reward/guru_reward.py"

# Run PPO training with configurations optimized for multi-domain training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    data.trust_remote_code=True \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.path="$MODEL_PATH" \
    critic.model.trust_remote_code=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=compute_score \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.type=adaptive \
    algorithm.kl_ctrl.kl_coef=0.02 \
    algorithm.kl_ctrl.target_kl=0.01 \
    trainer.critic_warmup=5 \
    trainer.logger='["console"]' \
    trainer.project_name='agentic_rl_scaling_law' \
    trainer.experiment_name='qwen2.5_3b_all_domains_ppo' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.total_epochs=3 $@