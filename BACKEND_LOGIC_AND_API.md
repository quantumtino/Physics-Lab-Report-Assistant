# Backend Logic, API, and Dependencies

This repository has been reduced to a backend-focused core.

## 1. Core Logic

### 1.1 LLM Interaction
- `LLMProcessor` initializes an OpenAI-compatible client targeting DashScope.
- It supports:
  - plain text completion: `generate_text(...)`
  - streaming completion: `generate_text_stream(...)`
  - OCR table extraction from image bytes: `extract_table_from_image_bytes(...)`
  - uncertainty-oriented conversational flow: `smart_uncertainty_conversation(...)`
  - formula normalization + uncertainty calculation pipeline: `compute_uncertainty_with_llm(...)`

### 1.2 Mathematical Computation
- `SymbolicMathTools` provides symbolic helpers:
  - partial derivative computation
  - expression evaluation
- `UncertaintyCalculator` performs analytical uncertainty propagation based on partial derivatives:
  - supports A-type and B-type uncertainty composition
  - computes total and relative uncertainty
  - calculates contribution share of each variable

### 1.3 Fitting Analysis
- `DataAnalyzer` provides:
  - `linear_fit(...)`: weighted/unweighted linear regression
  - `log_fit(...)`: weighted/unweighted logarithmic regression
  - `power_fit(...)`: weighted/unweighted power-law fitting on log-transformed data
  - `fourier_transform(...)`: FFT spectrum extraction

### 1.4 Plotting
- `DataAnalyzer` plotting methods return PNG Base64:
  - `plot_linear_fit(...)`
  - `plot_log_fit(...)`
  - `plot_power_fit(...)`
  - `plot_fourier_transform(...)`

## 2. Public Interfaces

All interfaces are in `backend_core.py`.

### 2.1 Symbolic Tools
- `call_tool(tool_name: str, **kwargs) -> Dict[str, Any]`

Supported `tool_name` values:
- `compute_partial_derivative`
- `evaluate_expression`

### 2.2 Uncertainty
- `UncertaintyCalculator.parse_formula(formula_str: str) -> sympy.Expr`
- `UncertaintyCalculator.compute_uncertainty_propagation_analytical(measurements: dict) -> dict`
- `validate_measurement_data(measurements: dict) -> (bool, list[str])`

Measurement schema example:
```python
{
  "m": {"value": 0.5, "unit": "kg", "a_uncertainty": 0.001, "b_uncertainty": 0.0005},
  "v": {"value": 2.3, "unit": "m/s", "a_uncertainty": 0.05, "b_uncertainty": 0.02}
}
```

### 2.3 Fitting & Plotting
- `DataAnalyzer.linear_fit(...)`
- `DataAnalyzer.log_fit(...)`
- `DataAnalyzer.power_fit(...)`
- `DataAnalyzer.fourier_transform(...)`
- `DataAnalyzer.plot_*` methods

### 2.4 LLM + OCR
- `LLMProcessor.generate_text(...)`
- `LLMProcessor.generate_text_stream(...)`
- `LLMProcessor.smart_uncertainty_conversation(...)`
- `LLMProcessor.compute_uncertainty_with_llm(...)`
- `LLMProcessor.extract_table_from_image_bytes(...)`

Environment variables:
- `DASHSCOPE_API_KEY` (required)
- `ALIBABA_CLOUD_MODEL` (optional)

## 3. Dependency List

Minimal runtime dependencies for `backend_core.py`:
- `numpy`
- `pandas`
- `matplotlib`
- `sympy`
- `python-dotenv`
- `openai`

## 4. Suggested Minimal requirements.txt

```txt
numpy
pandas
matplotlib
sympy
python-dotenv
openai
```
