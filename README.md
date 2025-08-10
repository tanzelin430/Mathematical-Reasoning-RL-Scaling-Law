# Agentic-RL-Scaling-Law

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Framework](https://img.shields.io/badge/framework-VeRL%20v0.3.1-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 Overview

This repository implements **Agentic RL Scaling Law Experiments** using the VeRL framework to explore how reinforcement learning performance scales with model size, data volume, and training steps across multiple domains (Math, Code, Logic, STEM) using the guru-RL-92k dataset.

### Key Features
- 🔧 **VeRL Framework Integration**: Leveraging VeRL's built-in reward system from Reasoning360
- 🎯 **Multi-Domain Support**: Math, Code, Logic, and STEM domains with automatic reward routing
- 📊 **Scaling Law Analysis**: Systematic exploration of model size, data, and compute scaling
- 🏃 **Production-Ready**: Battle-tested training pipelines with checkpoint management

## 🎯 Quick Start

### Minimal Setup
```bash
# Clone repository
git clone https://github.com/your-repo/Agentic-RL-Scaling-Law.git
cd Agentic-RL-Scaling-Law

# Install dependencies
pip install -r requirements.txt
cd verl/ && pip install -e . && cd ..

# Prepare data
python src/data/pre_verl.py

# Run training
bash scripts/train/run_ppo_qwen2.5_3b_verl_builtin.sh
```

### Training Examples

**Multi-domain training (all 4 domains):**
```bash
bash scripts/train/run_ppo_qwen2.5_3b_verl_builtin.sh
```

**Single-domain training (7B model):**
```bash
bash scripts/train/run_ppo_qwen2.5_7b_single_domain.sh
```

**Custom configuration:**
```bash
python3 -m verl.trainer.main_ppo \
    data.train_files="['/path/to/math.parquet']" \
    actor_rollout_ref.model.path="/path/to/model" \
    trainer.n_gpus_per_node=8
```

## 📁 Project Structure

```
Agentic-RL-Scaling-Law/
├── verl/                    # VeRL framework (from Reasoning360)
│   └── utils/
│       └── reward_score/    # Built-in reward scorers
├── src/
│   ├── data/               # Data preprocessing
│   └── reward/             # Legacy reward functions (deprecated)
├── scripts/
│   ├── train/              # Training scripts
│   └── train_data_check/   # Data validation tools
├── data/
│   └── guru_verl/          # Preprocessed guru-RL-92k dataset
│       ├── train/          # Training data by domain
│       ├── math/           # ~54.4k samples
│       ├── code/           # ~18k samples  
│       ├── logic/          # ~6.3k samples
│       └── stem/           # ~3.6k samples
└── results/                # Experiment outputs
    └── checkpoints/        # Model checkpoints
```

## 🧪 Experimental Setup

### Models
We experiment with models of various sizes to study scaling behaviors:
- **Qwen2.5 Series**: 3B, 7B, 14B, 32B
- **DeepSeek-R1-Distill-Qwen Series**: 1.5B, 7B, 14B
- **QwQ-32B**: For large-scale experiments

### Algorithms
- **PPO** (Proximal Policy Optimization)
- **GRPO** (Group Relative Policy Optimization)  
- **Reinforce++**

### Dataset
**[Guru-RL-92k](https://huggingface.co/datasets/LLM360/guru-RL-92k)**: A mixed-domain dataset with ~92k samples across:
- Math: 54.4k samples
- Code: 18k samples
- Logic: 6.3k samples
- STEM: 3.6k samples

### Metrics
- **Pass@1**: Average performance improvement over training steps
- **Pass@k Curves**: k∈[1,128] for measuring solution diversity
- **Compute Efficiency**: Performance gain per FLOP
- **Domain-Specific Accuracy**: Individual domain performance tracking

## 🔬 Research Focus

### Model Scale & Training Steps (N)
- Performance scaling with parameter count
- Model stability and overfitting analysis via Pass@k
- Sample efficiency across model sizes
- Training stability improvements with scale

### Data Scale & Proportions (D)
- RL data volume impact on performance ceiling
- SFT:RL data ratio optimization
- Marginal returns analysis
- Domain-specific data requirements

### Mixed-Domain Training
- Curriculum learning vs. mixed training
- Cross-domain transfer effectiveness
- Minimum data requirements for domain adaptation
- Model size impact on transfer learning

## 💻 Technical Implementation

### Environment Setup

```bash
# Install VeRL and dependencies
cd verl/
pip install -e .
pip3 install vllm==0.8.3
pip3 install flash-attn --no-build-isolation
```

### Data Preprocessing

Convert guru-RL-92k dataset to VeRL format:
```bash
# Preprocess all domains
python src/data/pre_verl.py

# Validate preprocessing
python scripts/train_data_check/check_data_sample.py
python scripts/train_data_check/detailed_reward_analysis.py
```

### VeRL Built-in Reward System

VeRL automatically routes reward computation based on the `data_source` field:

| Domain | Data Source Pattern | Scorer | Description |
|--------|-------------------|--------|-------------|
| **Math** | `math__*` | naive_dapo.py | Mathematical expression evaluation |
| **Code** | `codegen__*` | coder1 | Unit test execution with sandboxing |
| **Logic** | `logic__*` | arcagi.py | Pattern matching for logical reasoning |
| **STEM** | `stem__*` | stem scorer | Scientific problem evaluation |

**Key Advantages:**
- ✅ No custom reward function needed
- ✅ Automatic domain detection
- ✅ Battle-tested implementations
- ✅ Consistent scoring across domains

### Training Configuration

**Key Parameters:**
```yaml
# Model Configuration
actor_rollout_ref.model.path: "/path/to/Qwen2.5-3B"
actor_rollout_ref.model.lora_rank: 32
actor_rollout_ref.model.target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]

# Training Configuration
algorithm.adv_estimator: gae
algorithm.use_kl_in_reward: true
trainer.total_epochs: 3-5
trainer.n_gpus_per_node: 2-8

# Checkpoint Management
trainer.default_local_dir: results/checkpoints/${project}/${experiment}
trainer.save_freq: 10

# Logging
trainer.logger: ["console", "wandb"]
```

## 📊 Experiment Execution

### Running Training

**Basic Training:**
```bash
# Multi-domain training with all 4 domains
bash scripts/train/run_ppo_qwen2.5_3b_verl_builtin.sh

# Single-domain training (configure domain in script)
bash scripts/train/run_ppo_qwen2.5_7b_single_domain.sh
```

**Advanced Configuration:**
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['path/to/data.parquet']" \
    data.train_batch_size=128 \
    actor_rollout_ref.rollout.n=8 \
    trainer.total_epochs=5 \
    trainer.logger='["console", "wandb"]'
```

### Checkpoint Management

Checkpoints are automatically saved to:
```
results/
└── checkpoints/
    └── ${project_name}/
        └── ${experiment_name}/
            ├── epoch_0/
            ├── epoch_10/
            └── epoch_20/
```

Resume training from checkpoint:
```bash
trainer.resume_mode=auto  # Automatically find latest checkpoint
```

## 📈 Monitoring & Evaluation

### WandB Integration
```bash
# Enable WandB logging
wandb login
trainer.logger='["console", "wandb"]'
trainer.project_name='agentic_rl_scaling'
trainer.experiment_name='qwen3b_multi_domain'
```

### Results Organization
```
results/
├── checkpoints/     # Model checkpoints
├── logs/           # Training logs
└── metrics/        # Evaluation metrics
```

### Key Metrics Tracked
- **Training Metrics**: Loss, rewards, KL divergence
- **Performance Metrics**: Pass@1, Pass@k (k≤128)
- **Domain Breakdown**: Per-domain accuracy and improvements
- **Compute Efficiency**: FLOPs vs. performance curves

## 🔗 References & Acknowledgments

### Frameworks & Tools
- **VeRL Framework**: Advanced RL training framework from Reasoning360 project
- **vLLM**: High-performance inference engine
- **Flash Attention**: Memory-efficient attention implementation

### Datasets
- **Guru-RL-92k**: Multi-domain RL dataset ([HuggingFace](https://huggingface.co/datasets/LLM360/guru-RL-92k))

### Related Work
- [Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective](https://arxiv.org/pdf/2506.14965)
- Reasoning360 Project (VeRL source)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<<<<<<< HEAD
实验结果将自动保存到：
- `outputs/`: 训练日志按时间戳组织
- `results/`: 模型检查点、评估结果和分析图表
=======
**Note**: This is an active research project. For questions or collaboration, please open an issue or contact the maintainers.
>>>>>>> tzlexp
