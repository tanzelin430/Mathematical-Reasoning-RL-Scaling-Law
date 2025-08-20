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
├── scripts/
│   ├── train/              # Training scripts
│   └── train_data_check/   # Data validation tools
├── data/
│   └── guru_verl/          # Preprocessed guru-RL-92k dataset
│       ├── train/          # Training data by domain
│       ├── online_eval/    # val files
│       ├── balanced_samples/  # Training data balanced by domain
└── results/                # Experiment outputs
    └── checkpoints/        # Model checkpoints
```

## 🧪 Experimental Setup

### Models
We experiment with models of various sizes to study scaling behaviors:
- **Qwen2.5 Series**: 3B, 7B, 14B, 32B

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
pip install daytona_sdk
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
| **Math** | `math__*` | naive_dapo.py | Format check and answer check|
| **Code** | `codegen__*` | coder1 | Unit test execution with sandboxing |
| **Logic** | `logic__*` | arcagi.py | Pattern matching for logical reasoning |
| **STEM** | `stem__*` | stem scorer | Scientific problem evaluation |

**Key Advantages:**
- ✅ No custom reward function needed
- ✅ Automatic domain detection
- ✅ Battle-tested implementations
- ✅ Consistent scoring across domains

### Training Configuration

**Key Parameters for tuning:**
```yaml
train batch size: It refers to the number of prompts/queries processed in each rollout phase. the total number of trajectories (prompt-response pairs) is calculated as train_batch_size * rollout.n

mini batch size: It defines the number of prompts used in each parameter update step, with the corresponding number of trajectories being mini_batch_size * n

micro batch size: It controls the number of samples (or tokens, with use_dynamic_bsz=True) processed per GPU in forward/backward computations. In static mode, it directly affects GPU memory usage, while dynamic mode replaces it with token-based limits (e.g., *_max_token_len_per_gpu), optimizing for variable-length text.
```

## 🛠️ Developer Guide: Customizing Rewards & Prompts

### Code Execution Environment Setup

Now supports two code execution environments for code domain training:

**Setup:**
1. Get Daytona API key from [Daytona Platform](https://daytona.io)
2. Export in your training script:


### STEM

**Prerequisites:**
First, clone the STEM LLM Judge model:
```bash
git clone https://huggingface.co/TIGER-Lab/general-verifier
```

For STEM domain training, you have two options to run the training with LLM-as-a-Judge evaluation:

#### Method 1: Manual vLLM Server Setup

1. **Start the vLLM server manually** for STEM LLM Judge:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --host localhost \
     --port 8040 \
     --model /path/to/your/general-verifier \
     --served-model-name TIGER-Lab/general-verifier \
     --gpu-memory-utilization 0.7 \
     --max-model-len 512 \
     --tensor-parallel-size 1 \
     --trust-remote-code
   ```

2. **Run the training script**:
   ```bash
   bash scripts/train_yifan/run_grpo_qwen2.5_0.5b_single_domain.sh
   ```

#### Method 2: Automated Setup (Recommended)

Simply run the enhanced training script that automatically manages the vLLM server:

```bash
bash scripts/train_yifan/run_grpo_qwen2.5_0.5b_single_domain_new.sh
```

This script will:
- Automatically detect STEM domain training
- Start the vLLM server on a dedicated GPU
- Wait for the server to be ready
- Run the training with proper cleanup on exit

**Note**: The automated script reserves GPU 0 for the vLLM server and uses GPUs 1-7 for training to avoid resource conflicts.



### Modifying Reward Functions

**Key files for reward customization:**
- `verl/utils/reward_score/__init__.py` - Routing logic
- `verl/utils/reward_score/naive_dapo.py` - Math scorer  
- `verl/utils/reward_score/coder1.py` - Code scorer
- `verl/utils/reward_score/daytona.py` - Code Sandbox Scorer
- `verl/utils/reward_score/arcagi.py` - Logic scorer
- `verl/utils/reward_score/stem.py` - STEM scorer

**Example: Custom math reward (naive_dapo.py):**
```python
# Current 3-tier reward system
if correct:
    reward = 1.0
elif is_matched:  # Has \boxed{} format
    reward = -0.5  
else:
    reward = -1.0
```

### Modifying Prompts

**File to edit:** `src/data/pre_verl.py`

**Reference:** Reasoning360 project at `/root/work/Reasoning360/` contains excellent prompt templates with thinking tags.

**Testing your changes:**
```bash
python src/data/pre_verl.py  # Regenerate data
python scripts/train_data_check/check_data_sample.py  # Verify
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

