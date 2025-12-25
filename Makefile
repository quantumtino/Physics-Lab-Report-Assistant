.PHONY: help install install-python install-windows install-macos install-linux \
        run dev clean test update-deps show-help

# 默认目标
.DEFAULT_GOAL := help

# 虚拟环境路径
VENV := venv
PYTHON := python3
PYTHON_WIN := python

# 检查操作系统
UNAME_S := $(shell uname -s)

# 帮助文本
help:
	@echo "物理实验报告助手 - 项目管理命令"
	@echo ""
	@echo "使用方法: make [目标]"
	@echo ""
	@echo "可用目标:"
	@echo "  install              - 安装项目（自动检测平台）"
	@echo "  install-windows      - Windows 安装"
	@echo "  install-macos        - macOS 安装"
	@echo "  install-linux        - Linux 安装"
	@echo "  run                  - 运行应用"
	@echo "  dev                  - 开发模式运行"
	@echo "  clean                - 清理临时文件"
	@echo "  clean-venv           - 删除虚拟环境"
	@echo "  update-deps          - 更新依赖"
	@echo "  test                 - 运行测试"
	@echo "  show-help            - 显示此帮助"
	@echo ""
	@echo "快速开始:"
	@echo "  make install    # 首次安装"
	@echo "  make run        # 启动应用"
	@echo ""

# 安装目标（自动检测平台）
install: show-help
	@echo ""
	@echo "=========================================="
	@echo "物理实验报告助手 - 项目安装"
	@echo "=========================================="
	@echo ""
	@echo "检测到操作系统: $(UNAME_S)"
	@echo ""
	@echo "开始安装流程..."
	@echo ""
ifeq ($(UNAME_S),Linux)
	@make install-linux
else ifeq ($(UNAME_S),Darwin)
	@make install-macos
else
	@make install-windows
endif

# Windows 安装
install-windows:
	@echo "正在使用 Python 安装脚本..."
	@echo "运行 setup.py 进行环境配置..."
	@python setup.py

# macOS 安装
install-macos:
	@echo "正在为 macOS 配置环境..."
	@chmod +x setup.sh
	@./setup.sh

# Linux 安装
install-linux:
	@echo "正在为 Linux 配置环境..."
	@chmod +x setup.sh
	@./setup.sh

# 运行应用
run:
	@echo "启动 Streamlit 应用..."
	@echo "应用地址: http://localhost:8501"
	@echo ""
ifeq ($(UNAME_S),Linux)
	@bash run.sh
else ifeq ($(UNAME_S),Darwin)
	@bash run.sh
else
	@call run.bat
endif

# 开发模式
dev:
	@echo "进入开发模式..."
	@echo "启用热重载..."
	@streamlit run app.py --logger.level=debug

# 清理临时文件
clean:
	@echo "清理临时文件..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name ".DS_Store" -delete
	@echo "✓ 清理完成"

# 删除虚拟环境
clean-venv:
	@echo "删除虚�virtual环境..."
	@rm -rf $(VENV)
	@echo "✓ 虚拟环境已删除"

# 完全清理
clean-all: clean clean-venv
	@echo "✓ 完全清理完成"

# 更新依赖
update-deps:
	@echo "更新依赖包..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt --upgrade
	@echo "✓ 依赖更新完成"

# 运行测试
test:
	@echo "运行项目测试..."
	@if [ -f "testllm.py" ]; then \
		echo "找到测试文件: testllm.py"; \
		$(PYTHON) testllm.py; \
	else \
		echo "未找到测试文件"; \
	fi

# 显示帮助
show-help: help

# 检查 Python 版本
check-python:
	@echo "检查 Python 环境..."
	@$(PYTHON) --version
	@echo "pip 版本："
	@$(PYTHON) -m pip --version

# 显示依赖
show-deps:
	@echo "项目依赖列表:"
	@cat requirements.txt

# 虚拟环境状态
venv-status:
	@if [ -d "$(VENV)" ]; then \
		echo "✓ 虚拟环境存在"; \
		echo "路径: $(VENV)"; \
	else \
		echo "✗ 虚拟环境不存在"; \
		echo "运行 'make install' 来创建"; \
	fi

# 检查配置
check-config:
	@echo "检查项目配置..."
	@echo ""
	@echo "必要文件:"
	@ls -l app.py requirements.txt 2>/dev/null || echo "❌ 缺少关键文件"
	@echo ""
	@echo "配置文件:"
	@if [ -f ".env" ]; then \
		echo "✓ .env 文件存在"; \
	else \
		echo "⚠ .env 文件不存在"; \
	fi
	@echo ""

# 快速诊断
diagnose: check-python venv-status check-config
	@echo ""
	@echo "诊断完成"
	@echo "如有问题，请参考 INSTALL.md 或 QUICK_START.md"

# 显示项目信息
info:
	@echo "=========================================="
	@echo "物理实验报告助手 (Physics Lab Report Assistant)"
	@echo "=========================================="
	@echo ""
	@echo "项目文件:"
	@echo "  • app.py              - 主应用程序"
	@echo "  • analysis_module.py  - 数据分析模块"
	@echo "  • llm_integration.py  - LLM 集成模块"
	@echo "  • latex_generator.py  - LaTeX 生成模块"
	@echo ""
	@echo "安装脚本:"
	@echo "  • setup.py            - Python 跨平台安装脚本"
	@echo "  • setup.sh            - Linux/macOS Bash 脚本"
	@echo "  • setup.ps1           - Windows PowerShell 脚本"
	@echo ""
	@echo "快速启动:"
	@echo "  • run.sh              - Linux/macOS 启动脚本"
	@echo "  • run.bat             - Windows 启动脚本"
	@echo ""
	@echo "文档:"
	@echo "  • README.md           - 项目说明"
	@echo "  • INSTALL.md          - 安装指南"
	@echo "  • QUICK_START.md      - 快速开始"
	@echo "  • SCRIPTS.md          - 脚本说明"
	@echo "  • SETUP_SUMMARY.md    - 项目总结"
	@echo ""

# 更新 .env 文件配置
setup-env:
	@echo "配置 API-Key..."
	@echo "访问: https://dashscope.console.aliyun.com"
	@echo "复制 API-Key 后，按照脚本提示输入"
	@$(PYTHON) -c "from pathlib import Path; Path('.env').touch()"
	@$(PYTHON) -c "\
	import os; \
	api_key = input('请输入 DASHSCOPE_API_KEY: ').strip(); \
	if api_key: \
		with open('.env', 'w') as f: \
			f.write(f'DASHSCOPE_API_KEY={api_key}\n'); \
		print('✓ 配置已保存'); \
	else: \
		print('✗ API-Key 不能为空'); \
	"

# 显示快速命令
quick:
	@echo "常用命令速查:"
	@echo ""
	@echo "首次使用:"
	@echo "  make install      - 一键安装项目"
	@echo ""
	@echo "日常使用:"
	@echo "  make run          - 启动应用"
	@echo "  make dev          - 开发模式运行"
	@echo ""
	@echo "维护:"
	@echo "  make clean        - 清理临时文件"
	@echo "  make update-deps  - 更新依赖"
	@echo "  make diagnose     - 诊断环境问题"
	@echo ""
	@echo "更多信息:"
	@echo "  make help         - 显示完整帮助"
	@echo "  make info         - 显示项目信息"
	@echo ""
