#!/bin/bash
# Start SandboxFusion server locally

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SANDBOX_DIR="${SCRIPT_DIR}/SandboxFusion"
VENV_DIR="${SANDBOX_DIR}/venv"

# Default configuration
PORT=${SANDBOX_PORT:-8080}
HOST=${SANDBOX_HOST:-0.0.0.0}
WORKERS=${SANDBOX_WORKERS:-4}

echo "🚀 Starting SandboxFusion server..."

# Check if virtual environment exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "❌ Error: Virtual environment not found at ${VENV_DIR}"
    echo "Please run setup_local_sandbox.sh first"
    exit 1
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Navigate to SandboxFusion directory
cd "${SANDBOX_DIR}"

# Export environment variables for the server
export PYTHONPATH="${SANDBOX_DIR}:${PYTHONPATH}"

echo "📌 Server configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Workers: $WORKERS"
echo ""

# Start the server using uvicorn
echo "🌐 Starting server on http://${HOST}:${PORT}"
uvicorn sandbox.server.server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info