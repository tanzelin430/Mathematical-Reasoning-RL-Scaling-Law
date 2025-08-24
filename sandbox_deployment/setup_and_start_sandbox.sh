#!/bin/bash
# 一键安装并启动 SandboxFusion 服务

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SANDBOX_DIR="${SCRIPT_DIR}/SandboxFusion"
VENV_DIR="${SCRIPT_DIR}/sandboxfusion_venv"

echo "🚀 设置 SandboxFusion 服务..."

# 步骤 1: 创建并激活虚拟环境
echo "📦 步骤 1: 创建虚拟环境..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

# 步骤 2: 安装服务端依赖
echo "📦 步骤 2: 安装服务端依赖..."
pip install -r "${SCRIPT_DIR}/requirements_server.txt"

# 步骤 3: 检查并创建沙箱运行时环境
echo "📦 步骤 3: 设置沙箱运行时环境..."
cd "${SANDBOX_DIR}"

# 检查是否有 conda
if ! command -v conda &> /dev/null; then
    echo "⚠️  未找到 conda，需要先安装 miniconda"
    echo "   请运行: bash scripts/install-miniconda.sh"
    exit 1
fi

# 检查 sandbox-runtime 环境是否存在
if ! conda env list | grep -q "sandbox-runtime"; then
    echo "📦 创建 sandbox-runtime conda 环境..."
    cd runtime/python
    bash install-python-runtime.sh
    cd ../..
else
    echo "✅ sandbox-runtime 环境已存在"
fi

# 步骤 4: 启动服务
echo "🌐 步骤 4: 启动 SandboxFusion 服务..."
echo "📌 服务配置："
echo "   Host: 0.0.0.0"
echo "   Port: 8080"
echo "   API 文档: http://localhost:8080/docs"
echo ""
echo "📌 在你的训练脚本中设置："
echo "   export CODER1_EXEC=sandboxfusion"
echo "   export SANDBOX_FUSION_SERVERS=\"localhost:8080\""
echo ""

# 启动服务
cd "${SANDBOX_DIR}"
export PYTHONPATH="${SANDBOX_DIR}:${PYTHONPATH}"
uvicorn sandbox.server.server:app --host 0.0.0.0 --port 8080 --log-level info