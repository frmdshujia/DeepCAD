#!/bin/bash
# DeepCAD 一键启动脚本
# 用法: bash start.sh [cpu|cuda] [port]

set -e

DEVICE="${1:-cpu}"
PORT="${2:-8000}"
FRONTEND_PORT=8080

# ─── 自动寻找权重 ────────────────────────────────────────
CHECKPOINT_CANDIDATES=(
    "/data/home/shujia/CHD/result_retfound/basic_classify/finetune_train_UKB_SDPP_focal05_seed42_ep60/UKB_SDPP_focal05_seed42_ep60checkpoint-best.pth"
    "$(dirname "$0")/../../checkpoint-best.pth"
    "./checkpoint-best.pth"
)
CHECKPOINT=""
for c in "${CHECKPOINT_CANDIDATES[@]}"; do
    if [ -f "$c" ]; then
        CHECKPOINT="$c"
        break
    fi
done

if [ -z "$CHECKPOINT" ]; then
    echo "[ERROR] 找不到 checkpoint-best.pth，请手动指定路径后修改本脚本"
    exit 1
fi

echo "============================================"
echo "  DeepCAD 启动配置"
echo "============================================"
echo "  权重路径 : $CHECKPOINT"
echo "  推理设备 : $DEVICE"
echo "  后端端口 : $PORT"
echo "  前端端口 : $FRONTEND_PORT"
echo "============================================"

# ─── 启动后端 ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "[后端] 启动 Flask 服务..."
python backend.py \
    --host 0.0.0.0 \
    --port "$PORT" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" &
BACKEND_PID=$!
echo "  后端 PID: $BACKEND_PID"

# 等待后端就绪
sleep 3
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "[ERROR] 后端启动失败，请查看错误信息"
    exit 1
fi

# ─── 启动前端静态服务 ────────────────────────────────────
echo ""
echo "[前端] 启动静态文件服务..."
python3 -m http.server "$FRONTEND_PORT" &
FRONTEND_PID=$!
echo "  前端 PID: $FRONTEND_PID"

# ─── 显示访问地址 ────────────────────────────────────────
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "============================================"
echo "  服务已启动！"
echo "============================================"
echo "  前端页面  : http://${SERVER_IP}:${FRONTEND_PORT}"
echo "  后端 API  : http://${SERVER_IP}:${PORT}"
echo "  健康检查  : http://${SERVER_IP}:${PORT}/health"
echo ""
echo "  按 Ctrl+C 停止所有服务"
echo "============================================"
echo ""

# ─── 优雅退出 ────────────────────────────────────────────
trap "echo ''; echo '停止服务...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

wait
