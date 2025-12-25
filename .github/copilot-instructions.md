# Physics Experiment Report Assistant - Copilot Instructions

A Streamlit app ("物理实验报告助手") that generates physics experiment reports via LLM-based vision (OCR), data analysis, and report generation with Alibaba Qwen models.

## Architecture & Data Flow

**Three-stage workflow** (also the sidebar navigation):
1. **Vision/OCR Stage** (`ocr_page()`) → Upload images, use LLM vision to extract data tables
2. **Analysis Stage** (`analysis_page()`) → Fit models, compute uncertainties, visualize
3. **LLM Stage** (`llm_page_new()`) → AI-assisted report generation + LaTeX export

**State persistence**: All stages use `st.session_state` dict. Key fields:
- `ocr_dataframe`: Data table extracted via LLM vision from images (pandas DataFrame)
- `dataframe`: Final working data table (pandas DataFrame)
- `analysis_result`: Fit results (slope, intercept, uncertainties, R², χ²)
- `chat_history`: Multi-turn conversation with LLM for iterative report refinement
- `selected_model`: Active Qwen model variant (critical for feature availability)

**Module responsibility**:
- `analysis_module.py`: Physics math—linear/log/power fits, weighted least squares, uncertainty propagation, stat quantiles
- `llm_integration.py`: OpenAI-compatible API wrapper for Alibaba DashScope; handles vision, text generation, and deep thinking
- `latex_generator.py`: Jinja2-based `.tex` file assembly from analysis results + LLM text

## Critical Patterns

### Streamlit State & Navigation
- Initialize ALL `st.session_state` fields at top of `main()` BEFORE functions use them (lines 29–44 in app.py)
- Use `st.sidebar.radio()` for nav (lines 72–75); radio selection updates `st.session_state["current_page"]`
- Multi-page pattern: call functions like `ocr_page()`, `analysis_page()`, `llm_page_new()` conditionally

### LLM Model Selection & Capabilities (Critical Decision Point)
**Three Qwen model variants** available in UI (lines 837–849 in app.py):

| Model | ID | Best For | Speed | Accuracy | Cost | Key Features |
|-------|-----|----------|-------|----------|------|--------------|
| **Flash** | `qwen-flash-2025-07-28` | Quick prototyping, vision OCR | ⚡⚡⚡ Fast | Good | $ Cheap | Fast vision, suitable for table extraction |
| **Max** | `qwen3-max-2025-09-23` | Complex reasoning, high-quality report generation | ⚡ Normal | ⭐⭐⭐ Excellent | $$ | Best for detailed experimental analysis |
| **Plus** | `qwen-plus-2025-12-01` | Deep analysis, scientific rigor | ⚡ Slower | ⭐⭐⭐⭐ Best | $$$ | **Deep thinking** mode for complex physics concepts |

**Model-aware code patterns**:
- Line 981 in app.py: Automatically enables `enable_thinking=True` only for Plus model (`"plus" in st.session_state.selected_model`)
- `llm_integration.py` lines 73–79: Deep thinking passed as `extra_body` parameter in OpenAI API call
- **Recommendation**: Use Flash for OCR (fast table extraction), Max for multi-turn report generation, Plus when deep physical reasoning needed

### Vision/OCR Pattern (LLM-based)
- Method: `LLMProcessor.extract_table_from_image(image_path)` (see llm_integration.py)
- Returns pandas DataFrame with extracted table data
- Called in `ocr_page()` lines 99–103; results stored in `st.session_state.ocr_dataframe`
- User can edit extracted table before confirming (lines 110–114)
- **Error handling**: Check for empty DataFrame or "错误" column (lines 105–107)

### Uncertainty Calculations (Core Domain Logic)
- `DataAnalyzer.linear_fit()`: Implements weighted least squares when `y_err` provided; returns `(slope, intercept, R², slope_unc, intercept_unc, reduced_chi2)`
- `format_with_uncertainty(value, uncertainty, sig=2)`: Aligns decimal places—uncertainty drives precision
- All fits compute parameter uncertainties from covariance matrix; required by physics lab standards (GB/T 15000.2)

### LLM Integration (DashScope/OpenAI API)
- **Config**: API key from `DASHSCOPE_API_KEY` env var; base URL is `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Model selection**: `LLMProcessor.__init__(model=None)` accepts optional model param; defaults to `os.getenv("ALIBABA_CLOUD_MODEL", "qwen-flash")`
- **Anti-hallucination system prompt** (lines 31–35 in llm_integration.py): AI must ask clarifying questions if data incomplete; forbids fabrication
- **Streaming**: `chat_stream()` method yields chunks as `{"type": "thinking"|"content", "text": "..."}` dict (lines 981–993 in app.py)
- **Error handling**: Wrap API calls in try-except; handle missing API key at init time (line 23 in llm_integration.py)

### LaTeX Output
- `LatexGenerator` uses `jinja2.Template` with templates stored as class attributes (e.g., `self.report_template`)
- Template syntax: `{{ variable }}` for values, `{% for row in rows %}` for loops
- **LaTeX escaping**: Backslashes must be escaped as `\\` in Python strings; e.g., `\begin{table}` → `\\begin{table}`
- Call `render(ocr_text=..., dataframe=..., latex_tables=..., figures=...)` to generate final `.tex` content

## Development Workflow

**Setup & Run** (See [setup/SETUP_README.md](setup/SETUP_README.md) for details):
```bash
cd setup && python setup.py  # One-command install + auto-launch
streamlit run app.py          # Manual launch (requires .env with DASHSCOPE_API_KEY)
```

**Dependency tree**: numpy/scipy (math), pandas (data), matplotlib (plots), streamlit (UI), openai (LLM), jinja2 (templates), pillow (images)

## Common Tasks & Patterns

- **Add fitting model**: Extend `DataAnalyzer` with method like `def exponential_fit(...)`, return `(params..., uncertainties..., fit_quality)`. Then expose in `analysis_page()` UI via selectbox/button.
- **Modify LaTeX template**: Edit `LatexGenerator.report_template` string (lines ~40–100 in latex_generator.py); test with `render()` call in `llm_page_new()`.
- **New session state field**: Declare it in `main()` initialization block (lines 29–44) BEFORE any page function accesses it.
- **Handle LLM errors gracefully**: Catch `openai.APIError` and provide user-facing fallback message via `st.error()`.

## File Reference Map
- Entry: [app.py](app.py) (1216 lines; page routing, session init)
- Math engine: [analysis_module.py](analysis_module.py) (441 lines; DataAnalyzer class)
- LLM bridge: [llm_integration.py](llm_integration.py) (466 lines; LLMProcessor class)
- Report generation: [latex_generator.py](latex_generator.py) (156 lines; LatexGenerator class)
