#!/bin/bash

# 物理实验报告助手 - Linux/macOS 快速启动脚本
# 用于已配置好的环境快速启动应用

clear

echo ""
echo "=========================================="
echo "物理实验报告助手 - 快速启动"
echo "=========================================="
echo ""

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行 setup.sh 进行初始化"
    echo ""
    echo "运行以下命令初始化环境："
    echo "  ./setup.sh"
    exit 1
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "⚠️  .env 文件不存在，将使用默认配置"
    echo ""
fi

# 激活虚拟环境
echo "ℹ️  激活虚拟环境..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ 无法激活虚拟环境"
    exit 1
fi

echo "✓ 虚拟环境已激活"
echo ""

# 启动应用
echo "ℹ️  启动 Streamlit 应用..."
echo "ℹ️  应用地址: http://localhost:8501"
echo ""
echo "按 Ctrl+C 可以停止应用"
echo ""

streamlit run app.py

if [ $? -ne 0 ]; then
    echo "❌ 应用启动失败"
    exit 1
fi
