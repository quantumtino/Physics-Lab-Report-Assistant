@echo off
REM 物理实验报告助手 - Windows 快速启动脚本
REM 用于已配置好的环境快速启动应用

setlocal enabledelayedexpansion

cls
echo.
echo ========================================
echo 物理实验报告助手 - 快速启动
echo ========================================
echo.

REM 检查虚拟环境是否存在
if not exist "venv" (
    echo [错误] 虚拟环境不存在，请先运行 setup.ps1 进行初始化
    echo.
    echo 运行以下命令初始化环境：
    echo   .\setup.ps1
    pause
    exit /b 1
)

REM 检查 .env 文件
if not exist ".env" (
    echo [警告] .env 文件不存在，将使用默认配置
    echo.
)

REM 激活虚拟环境
echo [信息] 激活虚拟环境...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo [错误] 无法激活虚拟环境
    pause
    exit /b 1
)

echo [成功] 虚拟环境已激活
echo.

REM 启动应用
echo [信息] 启动 Streamlit 应用...
echo [信息] 应用地址: http://localhost:8501
echo.
echo 按 Ctrl+C 可以停止应用
echo.

streamlit run app.py

if errorlevel 1 (
    echo [错误] 应用启动失败
    pause
    exit /b 1
)

pause
