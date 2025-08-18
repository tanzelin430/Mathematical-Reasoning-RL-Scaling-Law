#!/bin/bash

# Deploy STEM LLM Judge Model
# This script starts a vllm server with the TIGER-Lab/general-verifier model
# for STEM domain verification

echo "Starting STEM LLM Judge deployment..."

# Check if vllm is installed
if ! pip show vllm > /dev/null 2>&1; then
    echo "vllm is not installed. Installing vllm..."
    pip install vllm==0.8.3
fi

# Set model path
MODEL_PATH="/home/local/PARTNERS/yz646/Agentic-RL-Scaling-Law/dev/general-verifier"
PORT=8000
GPU_MEMORY_UTILIZATION=0.15  # Adjust based on available GPU memory

# Kill any existing vllm process on the port
echo "Checking for existing vllm processes on port $PORT..."
lsof -ti:$PORT | xargs -r kill -9 2>/dev/null

# Start vllm server
echo "Starting vllm server with model at $MODEL_PATH on port $PORT..."
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"

# Export the URL for STEM judge
export STEM_LLM_JUDGE_URL="http://127.0.0.1:$PORT/v1/chat/completions"
echo "STEM_LLM_JUDGE_URL set to: $STEM_LLM_JUDGE_URL"

# Start the server in background
nohup vllm serve $MODEL_PATH \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --trust-remote-code \
    --max-seq-len 2048 \
    > vllm_stem_judge.log 2>&1 &

echo "vllm server PID: $!"
echo "Logs are being written to: vllm_stem_judge.log"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "✅ STEM LLM Judge server is ready!"
        echo ""
        echo "To use in training, make sure to export:"
        echo "export STEM_LLM_JUDGE_URL=\"http://127.0.0.1:$PORT/v1/chat/completions\""
        echo ""
        echo "To stop the server:"
        echo "kill $!"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "❌ Server failed to start. Check vllm_stem_judge.log for details."
exit 1