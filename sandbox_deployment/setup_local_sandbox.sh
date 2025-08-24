#!/bin/bash
# Setup SandboxFusion locally without Docker for offline deployment

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SANDBOX_DIR="${SCRIPT_DIR}/SandboxFusion"
VENV_DIR="${SANDBOX_DIR}/venv"

echo "🚀 Setting up SandboxFusion for local deployment (no Docker)..."

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "📌 Found Python version: $PYTHON_VERSION"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Install poetry in virtual environment
echo "📦 Installing poetry..."
pip install poetry==1.7.1

# Navigate to SandboxFusion directory
cd "${SANDBOX_DIR}"

# Modify pyproject.toml to support Python 3.10
echo "🔧 Modifying Python version requirement to support 3.10..."
sed -i 's/python = "^3.11"/python = "^3.10"/' pyproject.toml

# Install dependencies using poetry
echo "📦 Installing dependencies with poetry..."
poetry config virtualenvs.create false
poetry install --no-dev

# Install additional runtime dependencies for Python execution
echo "📦 Installing Python runtime dependencies..."
cd runtime/python
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Return to sandbox directory
cd "${SANDBOX_DIR}"

echo "✅ Local setup completed!"
echo ""
echo "📌 To download dependencies for offline deployment:"
echo "   mkdir -p ${SCRIPT_DIR}/offline_packages"
echo "   pip download -r <(poetry export -f requirements.txt) -d ${SCRIPT_DIR}/offline_packages/"
echo "   pip download poetry==1.7.1 -d ${SCRIPT_DIR}/offline_packages/"
echo ""
echo "📌 Next steps:"
echo "   1. Run ${SCRIPT_DIR}/start_sandbox_server.sh to start the server"
echo "   2. Configure CODER1_EXEC=sandboxfusion and SANDBOX_FUSION_SERVERS in your environment"