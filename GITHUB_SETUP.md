# 上传到 GitHub 指南

本项目已配置 git 仓库。按照以下步骤上传到 GitHub。

## 前置要求

1. **GitHub 账户**：访问 https://github.com 创建账户
2. **Git 已安装**：验证 `git --version`
3. **GitHub 认证**：配置 SSH 或 Personal Access Token

## 步骤 1：在 GitHub 创建新仓库

1. 登录 GitHub
2. 点击右上角 `+` 菜单 → `New repository`
3. 填写仓库信息：
   - **Repository name**: `physics-lab-report-assistant`
   - **Description**: `AI-powered physics experiment report assistant with OCR, data analysis and LLM collaboration`
   - **Visibility**: 选择 `Private`（推荐）或 `Public`
   - 其他选项保持默认（不要初始化README，我们已有）

4. 点击 `Create repository`

## 步骤 2：添加远程仓库

复制你的仓库 URL（使用 HTTPS 或 SSH），然后在本地执行：

```bash
cd "g:\课程\物理与人工智能\final_assignment"
git remote add origin https://github.com/YOUR_USERNAME/physics-lab-report-assistant.git
# 或使用 SSH（如果配置了）
# git remote add origin git@github.com:YOUR_USERNAME/physics-lab-report-assistant.git
```

## 步骤 3：推送代码

```bash
git branch -M main
git push -u origin main
```

## 隐私和安全措施

✅ **已实施的保护**：

1. **环境变量**（.env）：已在 .gitignore 中排除
   - 包含 DASHSCOPE_API_KEY（阿里云通义千问API密钥）
   - 绝不会被上传到 GitHub

2. **敏感文件**：
   - `__pycache__/` - Python 编译文件
   - `plots/` - 生成的图像（本地测试）
   - `.streamlit/secrets.toml` - Streamlit 密钥配置
   - `*.log` - 日志文件

3. **数据文件**：
   - CSV、Excel 等数据文件（如有敏感信息）
   - 临时文件和缓存

## 步骤 4：配置 GitHub Secrets（用于CI/CD）

如果计划使用 GitHub Actions，需要在 GitHub 上配置敏感信息：

1. 进入仓库 → Settings → Secrets and variables → Actions
2. 点击 `New repository secret`
3. 添加以下密钥：
   - Name: `DASHSCOPE_API_KEY`
   - Value: 你的阿里云 API 密钥

示例 GitHub Actions workflow（`.github/workflows/test.yml`）：

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      env:
        DASHSCOPE_API_KEY: ${{ secrets.DASHSCOPE_API_KEY }}
      run: python -m pytest tests/
```

## 步骤 5：.env 文件管理

**本地使用**（不上传）：

在本地创建 `.env` 文件：

```bash
# .env（本地，不上传）
DASHSCOPE_API_KEY=your_actual_api_key_here
ALIBABA_CLOUD_MODEL=qwen-max
```

**分享项目时**：

创建 `.env.example` 作为模板（可以上传）：

```bash
# .env.example（示例，可上传）
# 复制此文件为 .env 并填入你的实际 API 密钥
DASHSCOPE_API_KEY=your_dashscope_api_key
ALIBABA_CLOUD_MODEL=qwen-max
```

## 验证隐私设置

推送后，验证敏感文件未被上传：

```bash
# 查看仓库中的所有文件
git ls-files

# 确认没有以下文件：
# - .env（任何变体）
# - __pycache__/
# - plots/*.png
# - *.log
```

## 更新现有仓库

后续更新只需：

```bash
git add .
git commit -m "描述你的更改"
git push origin main
```

## 许可证建议

建议添加 LICENSE 文件。常见选择：

- **MIT License**：允许商业使用，宽松条件
- **Apache 2.0**：包含专利保护
- **GPL 3.0**：要求派生项目也开源

在仓库主页点击 "Add file" → "Create new file"，选择适当的许可证模板。

## 协作建议

如果需要多人开发：

1. 邀请成员：Settings → Collaborators → Add people
2. 建立分支工作流：
   - `main` 分支：稳定版本
   - `develop` 分支：开发版本
   - 功能分支：`feature/xxx`、`fix/xxx`

## 故障排除

**问题：推送被拒绝**

```bash
# 获取最新更改
git pull origin main --allow-unrelated-histories

# 重新推送
git push -u origin main
```

**问题：忘记添加 .env 到 .gitignore**

```bash
# 从 git 历史中移除文件（但保留本地副本）
git rm --cached .env

# 验证
git commit -m "Remove .env from tracking"
git push origin main
```

**问题：需要移除已上传的敏感信息**

参考：https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository

## 联系与支持

- **项目主页**：GitHub 仓库 URL
- **问题反馈**：使用 GitHub Issues
- **安全问题**：不要在 Issue 中泄露敏感信息，使用私密报告

---

**完成后，你的项目就已安全上传到 GitHub 了！**
