# 📦 安装与启动

完整的安装指南和常见问题解答

---

## 🚀 快速安装（推荐）

### 一条命令完成所有配置

```bash
python setup.py
```

脚本会自动处理：
- ✅ 检测 Python 版本（3.8+）
- ✅ 创建虚拟环境
- ✅ 安装所有依赖包
- ✅ 交互式配置 API-Key
- ✅ 验证环境
- ✅ 启动应用

---

## 📋 其他安装方式

### Windows 用户（PowerShell）

```powershell
.\setup.ps1
```

### macOS/Linux 用户（Bash）

```bash
chmod +x setup.sh
./setup.sh
```

---

## 🔑 配置 API-Key

### 自动配置（推荐）
运行 `python setup.py` 时会提示输入 API-Key

### 手动配置
在项目根目录创建 `.env` 文件：
```env
DASHSCOPE_API_KEY=你的_API_Key
```

### 获取 API-Key
1. 访问 https://dashscope.console.aliyun.com
2. 登录或注册阿里云账户（免费额度可用）
3. 在 API-Key 管理中创建新的 API-Key
4. 复制粘贴到脚本或 .env 文件

---

## ⚡ 快速启动（后续使用）

环境已配置后，快速启动应用：

**Windows**:
```bash
.\run.bat
```

**macOS/Linux**:
```bash
./run.sh
```

应用自动打开：http://localhost:8501

---

## 📋 前置要求

- Python 3.8+ - [下载](https://python.org)
- 网络连接 - 用于下载依赖和 LLM API

---

## 🔧 手动安装步骤

如果自动脚本出现问题，按以下步骤手动配置：

### 1. 创建虚拟环境
```bash
python -m venv venv
```

### 2. 激活虚拟环境
**Windows**: `.\venv\Scripts\Activate.ps1`
**Mac/Linux**: `source venv/bin/activate`

### 3. 安装依赖
```bash
pip install -r ../requirements.txt
```

### 4. 配置 API-Key
创建 `../.env` 文件：
```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 5. 启动应用
```bash
streamlit run ../app.py
```

---

## ❓ 常见问题

### Q: Python 未找到
**A**: 安装 [Python 3.8+](https://python.org)，确保 PATH 配置正确

### Q: 权限被拒绝（Linux/macOS）
**A**: 运行 `chmod +x setup.sh`

### Q: 依赖安装失败
**A**: 升级 pip：`pip install --upgrade pip`

### Q: API-Key 无效
**A**: 访问 DashScope 重新生成 API-Key

### Q: 虚拟环境问题
**A**: 删除 venv 文件夹，重新运行脚本

### Q: 应用无法启动
**A**: 检查 Python 版本和网络连接

---

## ✅ 验证安装成功

应该看到这些提示：
```
✓ Python 版本符合要求
✓ 虚拟环境已激活
✓ 依赖包安装完成
✓ .env 文件已创建
✓ 环境配置完成！
Streamlit 应用运行在 http://localhost:8501
```

---

**准备好了？** 运行 `python setup.py`

祝您使用愉快！🚀
