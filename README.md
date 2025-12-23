# 物理实验报告助手 (Physics Lab Report Assistant)

> **AI 驱动的物理实验数据分析与报告生成工具** - 集数据分析、科学拟合、AI 辅助撰写于一体

基于 **Streamlit + 通义千问 (Qwen LLM)** 的智能化物理实验报告助手，帮助学生和研究者快速、专业地完成实验数据处理与报告撰写。

---

## 🎯 核心创新点

### 1. **两阶段工作流**
- **阶段 1：数据分析与拟合** - 支持多种拟合模型、自动计算不确定度、生成可视化图表
- **阶段 2：AI 协作撰写** - 基于数据的对话式报告生成，遵循"禁止幻觉"原则

### 2. **学术严谨的不确定度体系** 
- **加权最小二乘法** - 按误差加权提升拟合精度
- **自动不确定度传播** - 参数不确定度、加权 R²、Reduced χ² 一次呈现
- **符合国标规范** - 遵循 GB/T 15000.2 物理量和单位标准

### 3. **AI 安全约束**
- **禁止幻觉机制** - AI 仅基于提供的数据撰写，遇信息不足主动提问
- **对话式规划** - Plan→Act 流程确保生成内容符合实验背景
- **片段化输出** - 报告分解为理论、分析、结论等可独立使用的模块

### 4. **工程化报告输出**
- **一键 ZIP 打包**，包含：
  - `report_segment.tex` - AI 生成的报告片段
  - `data_table.tex` - LaTeX 表格代码（可直接引用）
  - `plot.png` - 拟合图表（标准命名）
  - `data_table.csv` - 原始数据（便于追溯）
  - `custom_content.tex` - 用户自定义内容

---

## 📊 功能与特性

### **数据分析引擎** (`analysis_module.py`)

支持四种拟合模型：

| 拟合类型 | 公式 | 应用场景 | 输出 |
|---------|------|---------|------|
| **线性拟合** | $y = ax + b$ | 欧姆定律、弹簧周期 | $a±\Delta a$, $b±\Delta b$ |
| **对数拟合** | $y = a\ln(x) + b$ | 热容-温度关系 | $a$, $b$ 及不确定度 |
| **幂律拟合** | $y = Cx^k$ | 开普勒定律、黑体辐射 | $k±\Delta k$ |
| **FFT 频谱** | 傅里叶变换 | 信号处理、谐波分析 | 频率分量、幅度 |

**统计指标**：
- ✅ 加权 R²（加权决定系数）
- ✅ Reduced χ²（卡方统计量）
- ✅ 参数不确定度（自动计算）
- ✅ 误差传播（支持 x/y 方向误差棒）

### **AI 报告生成** (`llm_integration.py`)

**对话流程**：
1. **Plan 阶段** - AI 提问实验背景（名称、方法、物理规律等）
2. **Act 阶段** - 基于数据生成分析文段（理论、数据解读、结论讨论）
3. **输出** - LaTeX 格式，可直接嵌入学位论文

**模型配置**：
- 使用阿里云通义千问 (Qwen)
- 支持 `qwen-max`（高精度）和 `qwen-flash`（快速）
- OpenAI SDK 兼容接口

---

## 🏗️ 项目结构

```
final_assignment/
├── app.py                    # Streamlit 主应用（1215 行）
│   ├── ocr_page()           # 数据输入页面
│   ├── analysis_page()      # 数据分析页面
│   └── llm_page_new()       # 报告生成页面
│
├── analysis_module.py       # 数据分析引擎（441 行）
│   └── DataAnalyzer         # 拟合、不确定度、可视化
│
├── llm_integration.py       # LLM 集成（466 行）
│   └── LLMProcessor         # Qwen API、对话管理
│
├── requirements.txt         # 项目依赖
├── .env.example            # 环境变量模板
└── plots/                  # 输出图表目录（自动生成）
```

---

## ⚡ 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/quantumtino/Physics-Lab-Report-Assistant.git
cd final_assignment

# 安装依赖
pip install -r requirements.txt

# 配置 API 密钥
# 在 .env 文件中添加：
# DASHSCOPE_API_KEY=sk_xxxxxxxxxxxx
```

### 2. 启动应用

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

### 3. 使用流程

**示例：验证欧姆定律**

1️⃣ **数据输入**  
   - 上传 CSV 文件或粘贴数据表格
   - 选择 X 轴（电压 V）、Y 轴（电流 I）

2️⃣ **执行拟合**  
   - 选择拟合模型：线性拟合
   - 系统自动计算：$I = 0.2V + 0.01$，$R² = 0.9999$
   - 获取参数不确定度和拟合图表

3️⃣ **生成报告**  
   - Plan：AI 询问"是否考虑温度影响？"
   - Act：AI 生成报告片段（200-400 字）
   - 下载 ZIP 包，包含所有必要文件

---

## 💡 关键技术

### 加权最小二乘法
```
最小化：χ² = Σ wᵢ(yᵢ - ŷᵢ)²
权重：wᵢ = 1/σᵢ²（按测量误差逆平方加权）
```

### 不确定度传播
对于线性拟合参数 $a$，不确定度为：
$$\Delta a = \sqrt{\frac{1}{\sum w_i (x_i - \bar{x})²}}$$

### AI 上下文管理
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"实验：{context}"},
    {"role": "assistant", "content": plan_response},
    {"role": "user", "content": "现在撰写分析..."}
]
```

---

## 📈 性能对比

| 特性 | 本项目 | 传统手工 | 其他工具 |
|-----|--------|---------|---------|
| 数据输入 | ✅ CSV / 直接输入 | ❌ | ✅ |
| 加权拟合 | ✅ | ⚠️ 计算器 | ✅ |
| 不确定度 | ✅ 自动 | ❌ | ⚠️ 有限 |
| AI 报告 | ✅ 无幻觉 | ❌ | ⚠️ 可能出错 |
| LaTeX 输出 | ✅ 一键 | ❌ 手工 | ⚠️ 需调整 |

---

## 🔧 系统要求

- **Python** 3.8+
- **Streamlit** 1.30+
- **阿里云账户**（获取 DashScope API Key）

---

## 📚 文档

- [使用指南](USER_GUIDE.md) - 详细的逐步说明
- [改进日志](IMPROVEMENTS.md) - 版本更新记录

---

## 🤝 贡献

欢迎 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- **Streamlit** - Web UI 框架
- **通义千问 (Qwen)** - 大语言模型
- **NumPy/SciPy** - 科学计算
- **Matplotlib** - 数据可视化

---

**最后更新**: 2025年12月23日 | **当前版本**: 2.0 (精简版)
