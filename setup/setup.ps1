# Physics Lab Report Assistant - 安装脚本
# 功能：自动检测环境、配置依赖、获取API-key、启动应用

$ErrorActionPreference = "Stop"

# 颜色定义
function Write-Success {
    Write-Host "✓ $args" -ForegroundColor Green
}

function Write-Error-Custom {
    Write-Host "✗ $args" -ForegroundColor Red
}

function Write-Info {
    Write-Host "ℹ $args" -ForegroundColor Cyan
}

function Write-Warning-Custom {
    Write-Host "⚠ $args" -ForegroundColor Yellow
}

Write-Info "=========================================="
Write-Info "物理实验报告助手 - 环境配置脚本"
Write-Info "=========================================="
Write-Host ""

# 1. 检测 Python 版本
Write-Info "第一步：检测 Python 环境..."
try {
    $pythonVersion = python --version 2>&1
    Write-Success "检测到 Python: $pythonVersion"
} catch {
    Write-Error-Custom "未检测到 Python，请先安装 Python 3.8+"
    exit 1
}

# 检查 Python 版本是否符合要求
$versionMatch = $pythonVersion -match '(\d+\.\d+)'
if ($versionMatch) {
    $version = [float]$matches[1]
    if ($version -lt 3.8) {
        Write-Error-Custom "Python 版本过低，需要 3.8 或更高版本"
        exit 1
    }
    Write-Success "Python 版本符合要求"
} else {
    Write-Warning-Custom "无法解析 Python 版本信息"
}

Write-Host ""

# 2. 检测必要工具
Write-Info "第二步：检测必要工具..."
$missingTools = @()

# 检查 pip
try {
    $null = pip --version 2>&1
    Write-Success "pip 已安装"
} catch {
    $missingTools += "pip"
    Write-Error-Custom "pip 未检测到"
}

Write-Host ""

# 3. 检查 requirements.txt
Write-Info "第三步：检查依赖文件..."
$reqFile = "requirements.txt"
if (Test-Path $reqFile) {
    Write-Success "找到 $reqFile"
    $packages = Get-Content $reqFile | Where-Object { $_ -and -not $_.StartsWith("#") }
    Write-Info "将安装以下依赖包（共 $($packages.Count) 个）："
    $packages | ForEach-Object { Write-Host "  • $_" }
} else {
    Write-Error-Custom "未找到 requirements.txt"
    exit 1
}

Write-Host ""

# 4. 询问用户是否创建虚拟环境
Write-Info "第四步：虚拟环境配置..."
$venvName = "venv"
if (Test-Path $venvName) {
    Write-Warning-Custom "虚拟环境 '$venvName' 已存在"
    $useExisting = Read-Host "是否使用现有虚拟环境? (Y/n)"
    if ($useExisting -ne "n") {
        Write-Info "使用现有虚拟环境"
    } else {
        Write-Info "删除现有虚拟环境..."
        Remove-Item -Recurse -Force $venvName
        Write-Info "创建新的虚拟环境..."
        python -m venv $venvName
        Write-Success "虚拟环境创建完成"
    }
} else {
    Write-Info "创建虚拟环境 '$venvName'..."
    try {
        python -m venv $venvName
        Write-Success "虚拟环境创建成功"
    } catch {
        Write-Error-Custom "虚拟环境创建失败"
        exit 1
    }
}

# 激活虚拟环境
Write-Info "激活虚拟环境..."
$activateScript = ".\$venvName\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error-Custom "虚拟环境激活脚本未找到"
    exit 1
}
& $activateScript

Write-Success "虚拟环境已激活"
Write-Host ""

# 5. 安装依赖包
Write-Info "第五步：安装依赖包..."
try {
    Write-Host "安装过程中，请耐心等待..."
    pip install -r $reqFile -q
    Write-Success "依赖包安装完成"
} catch {
    Write-Error-Custom "依赖包安装失败: $_"
    exit 1
}

Write-Host ""

# 6. 检查并创建 .env 文件
Write-Info "第六步：API-Key 配置..."
$envFile = ".env"

if (Test-Path $envFile) {
    Write-Warning-Custom ".env 文件已存在"
    $apiKey = Select-String -Path $envFile -Pattern "DASHSCOPE_API_KEY" | Select-Object -First 1
    if ($apiKey) {
        Write-Success "检测到已配置的 API-Key"
    } else {
        Write-Warning-Custom ".env 文件中未找到 DASHSCOPE_API_KEY"
    }
    $updateKey = Read-Host "是否重新配置 API-Key? (y/N)"
    if ($updateKey -ne "y") {
        Write-Info "保持现有配置"
        Write-Host ""
        # 跳转到启动应用
        goto StartApp
    }
} else {
    Write-Info ".env 文件不存在，需要创建"
}

# 获取 API-Key
Write-Host ""
Write-Info "请按照以下步骤获取 API-Key："
Write-Host "  1. 访问阿里云 DashScope 平台: https://dashscope.console.aliyun.com"
Write-Host "  2. 登录您的阿里云账户"
Write-Host "  3. 创建或复制您的 API-Key"
Write-Host ""

$apiKey = Read-Host "请输入您的 DASHSCOPE_API_KEY"

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Error-Custom "API-Key 不能为空"
    exit 1
}

# 创建或更新 .env 文件
$envContent = @"
# DashScope API Configuration
DASHSCOPE_API_KEY=$apiKey

# (可选) 自定义模型，默认为 qwen-max
# ALIBABA_CLOUD_MODEL=qwen-max
"@

try {
    Set-Content -Path $envFile -Value $envContent
    Write-Success "✓ .env 文件已创建/更新"
} catch {
    Write-Error-Custom "无法写入 .env 文件: $_"
    exit 1
}

Write-Host ""

# 7. 验证环境配置
Write-Info "第七步：验证环境配置..."
Write-Info "检查必要包..."

$packagesToCheck = @(
    "streamlit",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "openai",
    "python-dotenv",
    "pillow"
)

$allInstalled = $true
foreach ($package in $packagesToCheck) {
    try {
        $null = python -c "import $($package -replace '-', '_')" 2>&1
        Write-Success "$package 已安装"
    } catch {
        Write-Error-Custom "$package 未安装或安装失败"
        $allInstalled = $false
    }
}

if (-not $allInstalled) {
    Write-Error-Custom "部分依赖包安装失败，请检查"
    exit 1
}

Write-Host ""
Write-Success "=========================================="
Write-Success "环境配置完成！"
Write-Success "=========================================="
Write-Host ""

# 8. 启动应用
:StartApp
Write-Info "第八步：启动应用..."
Write-Info "Streamlit 应用正在启动..."
Write-Info "应用地址: http://localhost:8501"
Write-Host ""

streamlit run app.py
