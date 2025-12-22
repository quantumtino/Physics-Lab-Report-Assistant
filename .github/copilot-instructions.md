# Physics Experiment Report Assistant - Copilot Instructions

This project is a Streamlit-based application ("物理实验报告助手") that assists in generating physics experiment reports using OCR, data analysis, and LLM capabilities.

## Architecture & Data Flow

- **Entry Point**: `app.py` is the main entry point. It uses `st.session_state` to manage state across different "pages" (simulated by functions like `ocr_page`, `analysis_page`).
- **Data Flow**:
  1.  **OCR**: Images are processed in `ocr_module.py`. Extracted text/tables are stored in `st.session_state['ocr_result']` or `st.session_state['dataframe']`.
  2.  **Analysis**: Data from session state is passed to `analysis_module.py` for fitting/plotting. Results go to `st.session_state['analysis_result']`.
  3.  **LLM**: `llm_integration.py` generates report text (theory, conclusion) using Alibaba Qwen models.
  4.  **LaTeX**: `latex_generator.py` combines all data into a final `.tex` report using Jinja2 templates.

## Key Components & Patterns

### 1. Streamlit UI (`app.py`)
- **State Management**: ALWAYS check and initialize `st.session_state` variables at the top of `app.py` before use.
- **Page Routing**: Use `st.sidebar.radio` to switch between functional views (`ocr_page`, `analysis_page`, etc.).
- **File Handling**: Use `tempfile` to handle uploaded files before passing paths to `OCRProcessor`.

### 2. LLM Integration (`llm_integration.py`)
- **Provider**: Uses Alibaba Cloud DashScope (Qwen) via the **OpenAI SDK** compatible endpoint.
- **Configuration**:
  - Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - API Key: `os.getenv("DASHSCOPE_API_KEY")`
  - Model: Defaults to `qwen-max` or `os.getenv("ALIBABA_CLOUD_MODEL")`.
- **Pattern**: Wrap API calls in `try-except` blocks to handle network/auth errors gracefully.

### 3. LaTeX Generation (`latex_generator.py`)
- **Templating**: Uses `jinja2.Template` for dynamic LaTeX generation.
- **Structure**: Templates are defined as string attributes in `LatexGenerator`.
- **Syntax**: Ensure double braces `{{ }}` are used for variables and `{% %}` for logic. Escape LaTeX backslashes properly in Python strings (e.g., `\\begin{table}`).

### 4. OCR & Analysis
- **OCR**: `OCRProcessor` (PaddleOCR) returns raw text or pandas DataFrames.
- **Analysis**: `DataAnalyzer` handles numpy/scipy calculations and matplotlib plotting.

## Development Workflow

- **Environment**: Requires `.env` file with `DASHSCOPE_API_KEY`.
- **Run Command**: `streamlit run app.py`
- **Dependencies**: Managed in `requirements.txt`. Key libs: `streamlit`, `paddleocr`, `openai`, `jinja2`.

## Common Tasks

- **Adding a new Analysis Method**:
  1.  Add method to `DataAnalyzer` class.
  2.  Update `analysis_page` in `app.py` to expose the option.
  3.  Store results in `st.session_state`.

- **Modifying Report Format**:
  1.  Edit the Jinja2 templates in `LatexGenerator`.
  2.  Ensure all new variables are passed during the `render` call.
