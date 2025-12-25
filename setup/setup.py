#!/usr/bin/env python3
"""
Physics Lab Report Assistant - 跨平台安装脚本
支持 Windows、macOS 和 Linux
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
from dotenv import load_dotenv

class Colors:
    """终端颜色定义"""
    SUCCESS = '\033[92m'
    ERROR = '\033[91m'
    WARNING = '\033[93m'
    INFO = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.SUCCESS}✓ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.ERROR}✗ {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.INFO}ℹ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.RESET}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.INFO}{'='*50}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.INFO}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.INFO}{'='*50}{Colors.RESET}\n")

def run_command(cmd, shell=False):
    """执行系统命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except FileNotFoundError:
        return False, f"命令未找到: {cmd[0]}"

def check_python_version():
    """检查 Python 版本"""
    print_info("第一步：检测 Python 环境...")
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    print_success(f"检测到 Python: {version_str}")
    
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 8):
        print_error("Python 版本过低，需要 3.8 或更高版本")
        sys.exit(1)
    
    print_success("Python 版本符合要求")

def check_pip():
    """检查 pip"""
    print_info("第二步：检查 pip...")
    success, output = run_command([sys.executable, "-m", "pip", "--version"])
    if success:
        print_success(output)
        return True
    else:
        print_error("pip 未找到或无法执行")
        return False

def check_requirements_file():
    """检查 requirements.txt 文件"""
    print_info("第三步：检查依赖文件...")
    req_file = Path("requirements.txt")
    
    if not req_file.exists():
        print_error("未找到 requirements.txt")
        sys.exit(1)
    
    print_success(f"找到 {req_file.name}")
    
    # 读取依赖列表
    packages = []
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                packages.append(line)
    
    print_info(f"将安装以下依赖包（共 {len(packages)} 个）：")
    for pkg in packages:
        print(f"  • {pkg}")
    
    return packages

def setup_venv(venv_name="venv"):
    """设置虚拟环境"""
    print_info("第四步：虚拟环境配置...")
    venv_path = Path(venv_name)
    
    if venv_path.exists():
        print_warning(f"虚拟环境 '{venv_name}' 已存在")
        response = input("是否使用现有虚拟环境? (Y/n): ").strip().lower()
        if response != "n":
            print_info("使用现有虚拟环境")
        else:
            print_info("删除现有虚拟环境...")
            import shutil
            shutil.rmtree(venv_path)
            print_info("创建新的虚拟环境...")
            create_new_venv(venv_name)
    else:
        print_info(f"创建虚拟环境 '{venv_name}'...")
        create_new_venv(venv_name)
    
    print_success("虚拟环境已准备就绪")
    return get_python_executable(venv_name)

def create_new_venv(venv_name):
    """创建新虚拟环境"""
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", venv_name],
            check=True
        )
        print_success("虚拟环境创建成功")
    except subprocess.CalledProcessError as e:
        print_error(f"虚拟环境创建失败: {e}")
        sys.exit(1)

def get_python_executable(venv_name):
    """获取虚拟环境中的 Python 可执行文件路径"""
    if platform.system() == "Windows":
        return str(Path(venv_name) / "Scripts" / "python.exe")
    else:
        return str(Path(venv_name) / "bin" / "python")

def install_dependencies(python_exe, packages):
    """安装依赖包"""
    print_info("第五步：安装依赖包...")
    print("安装过程中，请耐心等待...\n")
    
    try:
        subprocess.run(
            [python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print_success("依赖包安装完成")
    except subprocess.CalledProcessError as e:
        print_error(f"依赖包安装失败: {e}")
        sys.exit(1)

def setup_env_file():
    """设置 .env 文件"""
    print_info("第六步：API-Key 配置...")
    env_file = Path(".env")
    
    # 检查现有的 .env 文件
    api_key = None
    if env_file.exists():
        print_warning(".env 文件已存在")
        try:
            load_dotenv(env_file)
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if api_key:
                print_success("检测到已配置的 API-Key")
            else:
                print_warning(".env 文件中未找到 DASHSCOPE_API_KEY")
        except Exception as e:
            print_warning(f"读取 .env 文件失败: {e}")
        
        response = input("是否重新配置 API-Key? (y/N): ").strip().lower()
        if response != "y":
            print_info("保持现有配置")
            return
    
    # 获取新的 API-Key
    print()
    print_info("请按照以下步骤获取 API-Key：")
    print("  1. 访问阿里云 DashScope 平台: https://dashscope.console.aliyun.com")
    print("  2. 登录您的阿里云账户")
    print("  3. 创建或复制您的 API-Key")
    print()
    
    api_key = input("请输入您的 DASHSCOPE_API_KEY: ").strip()
    
    if not api_key:
        print_error("API-Key 不能为空")
        sys.exit(1)
    
    # 保存到 .env 文件
    env_content = f"""# DashScope API Configuration
DASHSCOPE_API_KEY={api_key}

# (可选) 自定义模型，默认为 qwen-max
# ALIBABA_CLOUD_MODEL=qwen-max
"""
    
    try:
        env_file.write_text(env_content)
        print_success(".env 文件已创建/更新")
    except Exception as e:
        print_error(f"无法写入 .env 文件: {e}")
        sys.exit(1)

def verify_packages(python_exe):
    """验证必要的包"""
    print_info("第七步：验证环境配置...")
    print_info("检查必要包...\n")
    
    packages_to_check = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("openai", "openai"),
        ("python-dotenv", "dotenv"),
        ("pillow", "PIL"),
    ]
    
    all_installed = True
    for pkg_name, import_name in packages_to_check:
        try:
            __import__(import_name)
            print_success(f"{pkg_name} 已安装")
        except ImportError:
            print_error(f"{pkg_name} 未安装或安装失败")
            all_installed = False
    
    if not all_installed:
        print_error("部分依赖包安装失败，请检查")
        sys.exit(1)

def start_app():
    """启动 Streamlit 应用"""
    print()
    print_header("环境配置完成！")
    print_info("第八步：启动应用...\n")
    print_info("Streamlit 应用正在启动...")
    print_info("应用地址: http://localhost:8501")
    print()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=False
        )
    except KeyboardInterrupt:
        print_info("\n应用已停止")
        sys.exit(0)
    except Exception as e:
        print_error(f"无法启动应用: {e}")
        sys.exit(1)

def main():
    """主函数"""
    print_header("物理实验报告助手 - 环境配置脚本")
    
    try:
        # 1. 检查 Python 版本
        check_python_version()
        print()
        
        # 2. 检查 pip
        if not check_pip():
            sys.exit(1)
        print()
        
        # 3. 检查 requirements.txt
        packages = check_requirements_file()
        print()
        
        # 4. 设置虚拟环境
        python_exe = setup_venv()
        print()
        
        # 5. 安装依赖
        install_dependencies(python_exe, packages)
        print()
        
        # 6. 设置 .env 文件
        setup_env_file()
        print()
        
        # 7. 验证包
        verify_packages(python_exe)
        print()
        
        # 8. 启动应用
        start_app()
        
    except KeyboardInterrupt:
        print_warning("\n用户中断安装")
        sys.exit(0)
    except Exception as e:
        print_error(f"发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
