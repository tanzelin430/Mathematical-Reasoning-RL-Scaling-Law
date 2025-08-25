#!/bin/bash

# 使用方法: ./start_sandboxes.sh 8
# 如果不提供参数，默认启动4个

NUM=$1
if [ -z "$NUM" ]; then
    NUM=8
fi

echo "启动 $NUM 个SandboxFusion实例..."

# 杀掉旧的
killall uvicorn 2>/dev/null
sleep 1
BASE_PORT=10086
# 启动新的
for i in $(seq 0 $((NUM-1))); do
    PORT=$((BASE_PORT + i))
    cd ../../SandboxFusion/ && PORT=$PORT make run-online > /tmp/sandbox_$PORT.log 2>&1 &
    echo "启动端口 $PORT (PID: $!, 日志: /tmp/sandbox_$PORT.log)"
    sleep 1
done

# 等待服务启动
echo ""
echo "等待服务启动..."
sleep 10

# 检查状态并测试执行
echo ""
echo "测试服务运行状态："
SUCCESS=0
for i in $(seq 0 $((NUM-1))); do
    PORT=$((BASE_PORT + i))
    echo -n "端口 $PORT: "
    
    # 测试代码执行
    RESPONSE=$(curl -s -X POST "http://localhost:$PORT/run_code" \
        -H "Content-Type: application/json" \
        -d '{"code": "print(\"Hello from port '$PORT'\")", "language": "python"}' 2>/dev/null)
    
    if echo "$RESPONSE" | grep -q '"status":"Success"'; then
        # 提取输出
        OUTPUT=$(echo "$RESPONSE" | grep -o '"stdout":"[^"]*"' | cut -d'"' -f4)
        echo "✅ 运行正常 (输出: $OUTPUT)"
        ((SUCCESS++))
    else
        echo "❌ 执行失败"
        if [ -n "$RESPONSE" ]; then
            echo "   响应: $RESPONSE"
        else
            echo "   服务未响应"
        fi
    fi
done

# 生成环境变量
SERVERS=""
for i in $(seq 0 $((NUM-1))); do
    PORT=$((8080 + i))
    [ -z "$SERVERS" ] && SERVERS="localhost:$PORT" || SERVERS="$SERVERS,localhost:$PORT"
done

echo ""
echo "=========================================="
echo "成功启动 $SUCCESS/$NUM 个实例"
echo "要关闭服务 请执行 pkill -f uvicorn"
echo ""
echo "=========================================="