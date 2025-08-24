# SandboxFusion 本地部署指南

## 架构说明

SandboxFusion 有两个独立的环境：
1. **服务端环境**：运行 FastAPI 服务，处理 HTTP 请求
2. **沙箱执行环境**：隔离的 conda 环境 `sandbox-runtime`，用于安全执行代码

两个环境完全独立，不会相互影响，也不会影响你的模型训练环境。

## 部署步骤

### 1. 准备服务端环境

```bash
cd /fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/sandbox_deployment

# 创建虚拟环境（推荐）
python3 -m venv sandboxfusion_venv
source sandboxfusion_venv/bin/activate

# 安装服务端依赖
pip install -r requirements_server.txt
```

### 2. 准备沙箱执行环境

```bash
cd SandboxFusion

# 安装 conda（如果还没有）
# bash scripts/install-miniconda.sh

# 创建独立的沙箱 conda 环境并安装依赖
cd runtime/python
bash install-python-runtime.sh

# 这会创建 sandbox-runtime 环境并安装 requirements.txt 中的所有包
```

### 3. 启动服务

```bash
cd /fs-computility/mabasic/tanzelin.p/work/Agentic-RL-Scaling-Law/sandbox_deployment
bash start_sandbox_server.sh

# 或者手动启动：
cd SandboxFusion
uvicorn sandbox.server.server:app --host 0.0.0.0 --port 8080
```

### 4. 配置 Coder1 使用 SandboxFusion

在你的训练脚本中设置环境变量：
```bash
export CODER1_EXEC=sandboxfusion
export SANDBOX_FUSION_SERVERS="localhost:8080"  # 或多个服务器 "server1:8080,server2:8080"
```

## 离线部署

如果目标机器没有网络，需要：

1. 在有网络的机器上下载所有依赖：
```bash
# 服务端依赖
pip download -r requirements_server.txt -d offline_packages/

# 沙箱运行时依赖
pip download -r SandboxFusion/runtime/python/requirements.txt -d offline_packages/
```

2. 在离线机器上安装：
```bash
# 服务端
pip install --no-index --find-links=offline_packages/ -r requirements_server.txt

# 沙箱环境
conda create -n sandbox-runtime python=3.10
conda activate sandbox-runtime
pip install --no-index --find-links=offline_packages/ -r requirements.txt
```

## 注意事项

- SandboxFusion 使用网络命名空间和 cgroup 进行隔离，可能需要 sudo 权限
- 沙箱环境完全独立，不会影响主环境
- 可以部署多个 SandboxFusion 实例进行负载均衡