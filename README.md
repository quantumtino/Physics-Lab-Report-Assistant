# Physics Lab Report Assistant

A desktop assistant for physics lab reporting with a 4-stage workflow:
1. OCR table extraction
2. Data analysis and plotting
3. Uncertainty analysis
4. AI chat and report drafting

Current implementation is Python GUI (`customtkinter`) with backend logic in `backend_core.py`.

## Features
- Extract table data from experiment images (OCR)
- Edit and save table data to project workspace
- Run linear/log/power fitting and FFT analysis
- Generate plots and analysis summaries
- Compute uncertainty propagation analytically
- Use LLM for uncertainty summary and report writing support
- Persist per-project artifacts by stage

## Project Layout

```text
.
|-- frontend_gui.py
|-- backend_core.py
|-- PROJECT_CONTEXT_FOR_AI.md
|-- BACKEND_LOGIC_AND_API.md
|-- WORK_PLAN.md
|-- requirements.txt
|-- environment.yml
|-- test/
```

Project runtime artifacts are organized as:

```text
<project_root>/
  project.json
  stage1/
    ocr_input/
    table.csv
    ocr_result.json
  stage2/
    analysis_input.csv
    analysis_result.json
    plots/
  stage3/
    uncertainty_input.json
    uncertainty_result.json
  stage4/
    chat_history.json
    draft.md
    draft.tex
```

## Requirements
- Python 3.11 recommended
- DashScope-compatible API key

Environment variables:
- `DASHSCOPE_API_KEY` (required)
- `ALIBABA_CLOUD_MODEL` (optional)

## Setup

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate physics-lab-assistant
```

### Option B: pip + venv

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python frontend_gui.py
```

On first launch:
- set API key in Settings
- select a project root folder
- create/open a project name

## Stage Workflow
- Stage 1 (OCR): import image, run OCR, edit table, save
- Stage 2 (Analysis): load CSV, choose method, fit/plot, save outputs
- Stage 3 (Uncertainty): input formula + measurements, compute propagation
- Stage 4 (AI Writing): chat with context, generate draft content, save history/draft

## Backend API Surface
Key classes and functions are in `backend_core.py`:
- `SymbolicMathTools`
- `UncertaintyCalculator`
- `DataAnalyzer`
- `LLMProcessor`
- `call_tool(...)`
- `validate_measurement_data(...)`

See `BACKEND_LOGIC_AND_API.md` for backend details.

## AI/Developer Context
Use `PROJECT_CONTEXT_FOR_AI.md` as the canonical AI-readable context file.
It explains:
- current architecture and source-of-truth files
- stage data contract
- roadmap relation with `WORK_PLAN.md`

## Git Strategy For Generated Files
This repository uses `.gitignore` to exclude:
- debug/cache/build artifacts
- local env and secrets
- AI workflow generated outputs under `test/**/stage*`

Keep source code and docs tracked, keep generated runtime artifacts untracked.

## Roadmap
`WORK_PLAN.md` documents a future Electron + React desktop frontend plan.
Current codebase remains Python GUI centric.
