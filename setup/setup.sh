#!/bin/bash

# Physics Lab Report Assistant - 安装脚本
# 支持 Linux 和 macOS

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 打印函数
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}==========================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}==========================================${NC}"
    echo ""
}

# 1. 检测 Python 版本
print_header "物理实验报告助手 - 环境配置脚本"

print_info "第一步：检测 Python 环境..."
if ! command -v python3 &> /dev/null; then
    print_error "未检测到 Python3，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "检测到 Python: $PYTHON_VERSION"

# 检查版本是否符合要求
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    print_error "Python 版本过低，需要 3.8 或更高版本"
    exit 1
fi

print_success "Python 版本符合要求"
echo ""

# 2. 检测 pip
print_info "第二步：检查 pip..."
if ! python3 -m pip --version &> /dev/null; then
    print_error "pip 未检测到"
    exit 1
fi
print_success "pip 已安装"
echo ""

# 3. 检查 requirements.txt
print_info "第三步：检查依赖文件..."
if [ ! -f "requirements.txt" ]; then
    print_error "未找到 requirements.txt"
    exit 1
fi

print_success "找到 requirements.txt"
print_info "将安装以下依赖包："
grep -v '^#' requirements.txt | grep -v '^$' | sed 's/^/  • /'
echo ""

# 4. 虚拟环境配置
print_info "第四步：虚拟环境配置..."
VENV_NAME="venv"

if [ -d "$VENV_NAME" ]; then
    print_warning "虚拟环境 '$VENV_NAME' 已存在"
    read -p "是否使用现有虚拟环境? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY != "n" ]]; then
        print_info "使用现有虚拟环境"
    else
        print_info "删除现有虚拟环境..."
        rm -rf "$VENV_NAME"
        print_info "创建新的虚拟环境..."
        python3 -m venv "$VENV_NAME"
        print_success "虚拟环境创建成功"
    fi
else
    print_info "创建虚拟环境 '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
    print_success "虚拟环境创建成功"
fi

# 激活虚拟环境
print_info "激活虚拟环境..."
source "$VENV_NAME/bin/activate"
print_success "虚拟环境已激活"
echo ""

# 5. 安装依赖包
print_info "第五步：安装依赖包..."
echo "安装过程中，请耐心等待..."
pip install -q -r requirements.txt
print_success "依赖包安装完成"
echo ""

# 6. 检查并创建 .env 文件
print_info "第六步：API-Key 配置..."
ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    print_warning ".env 文件已存在"
    if grep -q "DASHSCOPE_API_KEY" "$ENV_FILE"; then
        print_success "检测到已配置的 API-Key"
    else
        print_warning ".env 文件中未找到 DASHSCOPE_API_KEY"
    fi
    
    read -p "是否重新配置 API-Key? (y/N): " -n 1 -r
    echo
    if [[ $REPLY != "y" ]]; then
        print_info "保持现有配置"
        goto_start_app=1
    fi
fi

if [ "$goto_start_app" != "1" ]; then
    echo ""
    print_info "请按照以下步骤获取 API-Key："
    echo "  1. 访问阿里云 DashScope 平台: https://dashscope.console.aliyun.com"
    echo "  2. 登录您的阿里云账户"
    echo "  3. 创建或复制您的 API-Key"
    echo ""
    
    read -p "请输入您的 DASHSCOPE_API_KEY: " API_KEY
    
    if [ -z "$API_KEY" ]; then
        print_error "API-Key 不能为空"
        exit 1
    fi
    
    # 创建或更新 .env 文件
    cat > "$ENV_FILE" << EOF
# DashScope API Configuration
DASHSCOPE_API_KEY=$API_KEY

# (可选) 自定义模型，默认为 qwen-max
# ALIBABA_CLOUD_MODEL=qwen-max
EOF
    
    print_success ".env 文件已创建/更新"
fi

echo ""

# 7. 验证环境配置
print_info "第七步：验证环境配置..."
print_info "检查必要包..."
echo ""

PACKAGES=(
    "streamlit"
    "pandas"
    "numpy"
    "scipy"
    "matplotlib"
    "openai"
    "dotenv"
    "PIL"
)

PACKAGE_IMPORTS=(
    "streamlit"
    "pandas"
    "numpy"
    "scipy"
    "matplotlib"
    "openai"
    "dotenv"
    "PIL"
)

ALL_INSTALLED=true
for i in "${!PACKAGES[@]}"; do
    PACKAGE_NAME="${PACKAGES[$i]}"
    IMPORT_NAME="${PACKAGE_IMPORTS[$i]}"
    
    if python3 -c "import $IMPORT_NAME" 2>/dev/null; then
        print_success "$PACKAGE_NAME 已安装"
    else
        print_error "$PACKAGE_NAME 未安装或安装失败"
        ALL_INSTALLED=false
    fi
done

if [ "$ALL_INSTALLED" = false ]; then
    print_error "部分依赖包安装失败，请检查"
    exit 1
fi

echo ""

# 8. 启动应用
print_header "环境配置完成！"

print_info "第八步：启动应用..."
echo ""
print_info "Streamlit 应用正在启动..."
print_info "应用地址: http://localhost:8501"
echo ""

streamlit run app.py
