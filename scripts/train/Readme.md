# 个性化需要更改的参数
- SAMPLE_SIZE
- TRAINING_GPUS
- AUTHOR_NAME
- WANDB_DIR
- Required_Train_step
- WANDB_MODE(如果您的集群可以联网，请设置为online，否则设置为offline)

## 如果您的显存不够，请减小以下参数值
- train_prompt_bsz
- train_prompt_mini_bsz
- actor_seq_multiplier
- rollout_seq_multiplier
- gpu_memory_utilization（推荐0.5-0.7）

## 如果您发现您的GPU还有很多空余显存，请增大以下参数值
- train_prompt_bsz
- actor_seq_multiplier
- rollout_seq_multiplier

## 如果后续需要加上其他领域数据进行混合训练，或者需要改变训练领域
- 请修改src/data/sample_balanced_data.py中的domain_files字典，添加新的领域数据，并且修改samples_per_domain的值
- 请删除 data/guru_verl/train/balanced_samples 目录下的所有文件之后，再执行训练脚本

## 注意事项
- 训练过程中请用nvitop监控GPU利用率
- 如果对领域数量有修改，请删除 data/guru_verl/train/balanced_samples 目录下的所有文件之后，再执行训练脚本

## 固定的参数
Batchsize：512
Required_Train_step：1000

