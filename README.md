# Agentic-RL-Scaling-Law

[![Framework](https://img.shields.io/badge/framework-VeRL%20v0.3.1-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Exploring RL scaling laws across Math, Code, Logic, and STEM domains using VeRL framework.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and install
Basic Info: Python 3.12 Cuda 12.4 torch 2.6.0
git clone https://github.com/your-repo/Agentic-RL-Scaling-Law.git
cd Agentic-RL-Scaling-Law
pip install -r requirements.txt
pip install -e verl[gpu,test,math,vllm,code]

# Prepare data
python src/data/pre_verl.py
```

### 2. Code Execution Setup (SandboxFusion)

```bash
# Install SandboxFusion
git clone https://github.com/bytedance/SandboxFusion.git 
cd SandboxFusion
poetry install
mkdir -p docs/build
cd runtime/python && bash install-python-runtime.sh && cd ../..

# Start multiple instances for training
cd ../
bash scripts/sandbox/start_sandboxes.sh 8

# Copy the export command from output:
export SANDBOX_FUSION_SERVERS="localhost:8080,localhost:8081,..."
```

### 3. Run Training

```bash
# Single domain (math/code/logic/stem)
bash scripts/train/run_ppo_qwen2.5_7b_single_domain.sh

# Mixed domains
bash scripts/train/run_grpo_qwen2.5_7b_mixed_domain.sh

# With SandboxFusion for code domain
export CODER1_EXEC=sandboxfusion
bash scripts/train/run_ppo_qwen2.5_7b_sandboxfusion.sh
```

## 📁 Project Structure

```
Agentic-RL-Scaling-Law/
├── verl/                    # VeRL framework with built-in reward scorers
│   └── utils/
│       └── reward_score/    
│           ├── coder1/      # Code execution with SandboxFusion
│           ├── naive_dapo.py # Math scorer
│           ├── arcagi.py    # Logic scorer
│           └── stem.py      # STEM scorer
├── src/
│   └── data/               # Data preprocessing
├── scripts/
│   ├── train/              # Training scripts
│   └── sandbox/            # SandboxFusion management
├── data/
│   └── guru_verl/          # Preprocessed data (~92k samples)
└── results/                # Checkpoints and logs
```

## 🔧 Key Features

- **Automatic reward routing** based on data_source field
- **Secure code execution** with SandboxFusion (4% overhead vs subprocess)
- **Multi-domain support** with VeRL's battle-tested scorers
- **Dynamic batching** for efficient GPU utilization

## 🐛 Troubleshooting

- **SandboxFusion not responding**: Check ports 8080-808X are free
- **OOM errors**: Reduce batch size or use gradient checkpointing
- **Code format issues**: Model must output code in \`\`\`python blocks

## 🔗 Acknowledgments

- **VeRL Framework** from Reasoning360 project
- **SandboxFusion** by ByteDance
- **Guru-RL-92k Dataset** from [HuggingFace](https://huggingface.co/datasets/LLM360/guru-RL-92k)

## 📝 License

MIT License - see [LICENSE](LICENSE) file.