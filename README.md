# 🔬 物理实验报告助手

> **AI 驱动的物理实验数据分析与报告生成工具**

集数据分析、科学拟合、AI 辅助撰写于一体。基于 **Streamlit + 通义千问 (Qwen LLM)**

---

## 🚀 快速开始

### 安装（一条命令）

```bash
cd setup
python setup.py
```

详细说明请查看 [setup/SETUP_README.md](setup/SETUP_README.md)

### 启动应用

安装完成后自动启动，或访问：http://localhost:8501

---

## 🎯 核心特性

### 1. **两阶段工作流**
- 🔢 **数据分析与拟合** - 支持多种拟合模型、自动计算不确定度、生成可视化
- ✍️ **AI 协作撰写** - 基于数据生成专业报告文段，遵循学术规范

### 2. **学术严谨的不确定度体系**
- 加权最小二乘法提升拟合精度
- 自动计算参数不确定度、加权 R²、χ² 等统计量
- 符合 GB/T 15000.2 物理量和单位标准

### 3. **AI 安全约束**
- 禁止幻觉机制 - AI 仅基于提供的数据撰写
- 对话式规划 - 确保生成内容符合实验背景
- 片段化输出 - 报告分解为可独立使用的模块

### 4. **工程化报告输出**
- 一键生成 ZIP 包含：LaTeX 报告、数据表、图表、CSV 数据

---

## 📊 功能模块

### 数据分析引擎
支持 4 种拟合模型：线性拟合、对数拟合、幂律拟合、FFT 频谱分析
自动计算不确定度、拟合质量指标、误差传播

### LLM 报告生成
基于阿里云通义千问模型，支持对话式规划和 AI 辅助撰写
生成 LaTeX 格式报告，可直接用于学位论文

### 数据可视化
matplotlib 绘制高质量图表
支持误差棒、拟合曲线、参数展示

---

## 🛠️ 环境要求

- **Python 3.8+**
- **网络连接** - 用于下载依赖和访问 LLM API
- **阿里云账户** - 获取 DashScope API-Key（免费额度可用）

---

## 📚 文档导航

| 文件 | 说明 |
|------|------|
| **[setup/SETUP_README.md](setup/SETUP_README.md)** | ⚡ 快速安装指南和常见问题 |
| **[setup/setup.py](setup/setup.py)** | 🐍 推荐：跨平台脚本 |

---

## 💡 使用流程

1. **上传数据** - 支持图像 OCR 或直接输入数据
2. **分析拟合** - 选择拟合模型、查看结果和不确定度
3. **生成报告** - 与 AI 对话生成实验报告文段

---

## 🏗️ 项目结构

```
Physics-Lab-Report-Assistant/
├── setup/              # 📦 安装脚本和文档
├── app.py              # 🎨 Streamlit 主应用
├── analysis_module.py  # 📊 数据分析引擎
├── llm_integration.py  # 🤖 LLM 集成
├── latex_generator.py  # 📄 LaTeX 生成
└── requirements.txt    # 依赖列表
```

---

## 🔧 配置

### API-Key

1. 访问 https://dashscope.console.aliyun.com
2. 创建 API-Key（免费额度可用）
3. 运行脚本时输入，或在 `.env` 文件中配置：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

---

## ❓ 常见问题

**Q: 需要多长时间安装？**
A: 首次 5-10 分钟，之后每次启动 10 秒

**Q: 支持哪些系统？**
A: Windows、macOS、Linux

**Q: 可以离线使用吗？**
A: 数据分析可以，报告生成需要网络

**更多问题？** 查看 [setup/README.md](setup/README.md)

---

## 🙏 致谢

- **Streamlit** - Web UI 框架
- **阿里云通义千问** - 大语言模型
- **Copilot** - 代码生成辅助

---

**版本**: 2.0 | **更新**: 2025年12月25日
