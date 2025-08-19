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
HOST="localhost"
PORT=8000
MODEL_PATH="/fs-computility/mabasic/shared/models/general-verifier"
SERVED_MODEL_NAME="TIGER-Lab/general-verifier"
GPU_MEMORY_UTILIZATION=0.15  # Adjust based on available GPU memory
MAX_MODEL_LEN=2048
TENSOR_PARALLEL_SIZE=1

# Start vllm server
echo "Starting vllm server with model at $MODEL_PATH on port $PORT..."
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"

# Export the URL for STEM judge
export STEM_LLM_JUDGE_URL="http://$HOST:$PORT/v1/chat/completions"
echo "STEM_LLM_JUDGE_URL set to: $STEM_LLM_JUDGE_URL"

# Start the server in background
nohup python -m vllm.entrypoints.openai.api_server \
    --host $HOST \
    --port $PORT \
    --model $MODEL_PATH \
    --served-model-name $SERVED_MODEL_NAME \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --trust-remote-code > outputs/vllm_stem_judge.log 2>&1 &

echo "vllm server PID: $!"
echo "Logs are being written to: outputs/vllm_stem_judge.log"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://$HOST:$PORT/health > /dev/null 2>&1; then
        echo "✅ STEM LLM Judge server is ready!"
        echo ""
        echo "To use in training, make sure to export:"
        echo "export STEM_LLM_JUDGE_URL=\"http://$HOST:$PORT/v1/chat/completions\""
        echo ""
        echo "To stop the server:"
        echo "kill $!"
        exit 0
    fi
    echo -n "."
    sleep 3
done

echo ""
echo "❌ Server failed to start. Check vllm_stem_judge.log for details."
exit 1
