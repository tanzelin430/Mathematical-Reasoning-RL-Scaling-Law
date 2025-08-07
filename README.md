# Agentic-RL-Scaling-Law
Agentic RL Scaling Law Experiments

## 实验设置

### 模型
我们的实验需要囊括各个参数量的模型，可从以下具有深度推理能力的基座模型中选择：
- DeepSeek-R1-Distill-Qwen系列：1.5B、7B、14B
- QwQ-32B

### 基础框架
VeRL

### 算法
PPO GRPO Reinforce++
### 数据集
数据集：[Guru-RL-92k](https://huggingface.co/datasets/LLM360/guru-RL-92k) 该数据集是一个混合数据集，本研究用该实验作为初始实验数据集有两大原因，一是保证各分工组的实验数据集和Code-Base保持一致，便于项目推进和讨论；二是在Scaling Law探究的初期可以尽量做出一个General的结论，而避免谈论具体的Domain。在实验后期，我们会做General Scaling Law->Domain-Specific Scaling Law的对比与迁移。同时由于该数据集不含SFT数据，相关数据若需使用需要人工合成。

### Metric
- Pass@1 平均性能提升（横坐标训练步数，纵坐标Pass@1）
- Pass@k曲线（横坐标k，纵坐标Pass@k，在不同的训练步数下放置多张图）
  - 将k=128单独列出来，形成一个“创造性指数”，我们认为在这个k值下面模型已经sample出了它能找到的所有解
  - 该指标用于评测模型是否过拟合到了奖励信号上
  - k_max = 128
- 每单位计算量的性能提升：计算量增加的Flops与Pass@1的上升之间的关系

### 探究点
在实验中，我们遵循控制变量法这一基础原则，我们主要有三大主要变量：模型规模、数据量、训练步数
重点关注前三点


**模型规模与训练步数(N)**【Zelin、chenzhang、zaibin(literature review)】
- 随着参数量增加，使用相同Setting进行RFT（Reinforcement Learning Fine-tuning）之后模型性能是否提升（Pass@1）
- 参数量对于模型稳定性影响：RFT之后，观察模型是否过拟合到奖励模型（观察Pass@K）
- 参数量对于样本效率的影响：随着参数量提升，是否需要更多的训练步数才能让模型收敛（或者说，达到最大性能）
- 更大的模型规模是否可以带来更加稳定的训练过程
- 训练步数与过拟合到奖励信号、以及泛化性之间的关系：在训练过程中如果需要保持模型的泛化性能，是否需要early stop（观察Pass@k与训练步数之间的关系）


**数据规模与比例(D)**【周恒、钰涛、zelin】
- RFT过程中的样本数量记为$$D_{RL}$$，对于相同的数据集，更多的$$D_{RL}$$是否可以持续提升模型性能上限，性能上限何时出现（Pass@1），是否存在边际收益递减的情况？
- SFT数据与RFT数据的“规模匹配”问题：通过调整$$
D_{SFT}:D_{RL}$$，观察RFT过程中：1.模型性能的变化Scaling Law是否与上一点中提到的一致（比如，收敛到性能上限的速度，观察曲线斜率，SFT 性能随数据规模的提升是否会使 RL 微调的缩放斜率更陡峭）2. 对于性能上限的影响（Pass@1）


**混合数据训练**【Xiangyuan Yifan】
- 对数据进行难度分级，观察由浅入深地分级进行训练和混合不同难度数据进行训练的收敛曲线是否一致（课程学习or直接开训）
- 对于混合数据的跨领域训练，观察在新领域能够实现Domain-transfer的最小数据量要求，或者说能否有效地实现Domain Transfer。
- 若Domain-Transfer是有效的，则需要探究：
  - 是否越大参数量的模型，在新领域上强化学习效率更高，而参数量少的模型反而难以泛化【待观察实际效果】


  
**奖励模型（RM）的规模与质量**【Preference Based Only，想法待完善】
- RM 的规模与 RL 微调性能的关系：相同数据微调出的更大的 RM 是否能更精准地引导 RL，使主模型性能随 RM规模提升而提升？
- RM 的数据规模（$$D_{RM}$$）：当$$D_{RM}$$增加时，RL 微调的性能是否遵循特定规律？是否存在 “RM 数据瓶颈”（即 RM 的数据不足会限制主模型的缩放潜力）？


**奖励规则的复杂度**【Rule Based Only，想法待完善】
- 规则复杂度与模型规模的交互：更大规模的模型是否更能理解规则的深层逻辑？此时 scaling law 的斜率是否更陡峭（即模型规模对性能的边际效益更高）
- 与 Preferenced RM 的对比：相同模型规模下，rule-based RM 的 scaling law 是否比 learned RM（基于数据训练的 RM）更早出现 “平台期”？

  
**Long Horizon Task多轮交互次数**【想法待完善】
依赖的数据集：https://github.com/abdulhaim/LMRL-Gym



## 实验方法
- 上述探究点需要在各领域数据集上利用不同的强化学习算法（PPO，GRPO，Reinforce++, ARPO）进行操作，观察Scaling Law在不同领域的变化情况。
- 由于Guru-RL-92k数据集是一个纯RL数据集，目前打算由我、周恒、钰涛一起进行SFT数据合成。
- 前三个探究点应该一起进行，具体负责人已经列在相关探究点之后，具体而言：
  - 每个研究点都对应框架的不同功能，三个研究点都需要框架基础的RFT能力，例如第二个研究点需要在基础框架中引入SFT能力，需要在数据集中人工制造SFT数据。在实际推进过程中，每个方向都应该有一个branch，在周会前进行CodeReview。
  - Literature Review应该在实验过程中同步进行，主要是为了收集近期相关工作并且cover在自己所研究的方向中。
  - 混合数据训练应该以数据集对应paper：[Revisiting Reinforcement Learning for LLM Reasoning
from A Cross-Domain Perspective](https://arxiv.org/pdf/2506.14965)的相关settings继续进行。
  - 相关实验数据请统一存在[Google Excel](https://docs.google.com/spreadsheets/d/1fRCf3vYXwccsNcc5z_T6-jU20ErIVYOUCg7W_XfvRVs/edit?usp=sharing)中

## 实验实现细节

### 如何运行实验

#### 环境配置
1. 克隆代码仓库：
```bash
git clone --recursive https://github.com/your-repo/Agentic-RL-Scaling-Law.git
cd Agentic-RL-Scaling-Law
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置VeRL框架：
```bash
# 安装VeRL
cd verl/
pip install -e .
# Install the latest stable version of vLLM
pip3 install vllm==0.8.3
# Install flash-attn
pip3 install flash-attn --no-build-isolation


```

#### 数据预处理
**目的:** 将guru-RL-92k数据集转换为VeRL框架格式，涵盖Math、Code、Logic、STEM四个领域

**运行预处理:**
```bash
python src/data/pre_verl.py
```

**验证预处理结果:**
```bash
# 检查数据格式
python scripts/train_data_check/check_data_sample.py

# 详细奖励分析
python scripts/train_data_check/detailed_reward_analysis.py
```
#### 奖励计算方法
各领域采用不同的奖励计算策略，确保评估的准确性和领域特异性：

| 领域 | 计算方法 | 特点 | 评分范围 |
|------|----------|------|----------|
| **Math** | VeRL内置math_score + 模式匹配 | 处理boxed答案和数学表达式 | 0.0-1.0 |
| **Code** | 单元测试执行 + 通过率计算 | 安全代码执行环境，真实测试运行 | 0.0-1.0 (梯度评分) |
| **Logic** | 规则模式匹配 | 是/否答案标准化 | 0.0-1.0 |
| **STEM** | Math scorer + 模式匹配 | 数值问题和描述性答案处理 | 0.0-1.0 |

#### 运行训练实验

**单领域训练 (Math+STEM):**
```bash
bash scripts/train/run_ppo_qwen2.5_3b_guru.sh
```

**四领域混合训练:**
```bash
bash scripts/train/run_ppo_qwen2.5_3b_all_domains.sh
```

#### 重要配置选项
训练脚本支持以下关键配置的自定义：

**数据混洗:** `data.shuffle=True/False` - 控制是否随机打乱训练数据

**日志设置:**
- `trainer.logger='["console"]'` - 仅控制台输出
- `trainer.logger='["console", "wandb"]'` - 控制台 + WandB记录 (需要WandB认证)

### 项目成果
- ✅ **四领域混合训练pipeline完全可用** - Math, Code, Logic, STEM
- ✅ **所有领域奖励函数实现并验证** - 包括Code领域单元测试执行突破
- ✅ **Repository结构优化完成** - 脚本按功能组织
- ✅ **WandB集成问题解决** - 支持灵活的日志配置

### 实验监控与结果
实验结果将自动保存到：
- `outputs/`: 训练日志按时间戳组织
- `results/`: 模型检查点、评估结果和分析图表
- WandB集成：支持实时训练监控和指标可视化
