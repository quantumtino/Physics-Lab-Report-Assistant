# JS Desktop Frontend Work Plan

## 1. Goal
- Rebuild the product as a JavaScript desktop application (not Streamlit web UI).
- Keep current backend capability from `backend_core.py` and expose it to the desktop frontend.
- Produce a Windows `.exe` distribution build.

## 2. Product Requirements Mapping

### 2.1 First-time API key flow
- On first launch, prompt user to input API key.
- Persist API key in local config.
- Auto-fill/reuse API key on next launch.
- Allow API key modification in Settings.

### 2.2 Project workspace flow
- User selects a folder and names a project for each new working session.
- The selected folder is the single source of truth for all generated artifacts.

### 2.3 Four-stage workflow in main UI
1. Table recognition
2. Data analysis and plotting
3. Uncertainty analysis
4. AI chat and writing

### 2.4 Inter-stage data handoff
- Support one-click import from previous stage.
- Support manual upload/import at each stage as fallback.

### 2.5 Persistent outputs per project folder
- Table data
- Plot images
- Uncertainty analysis results
- AI conversation history
- Writing fragments/drafts

### 2.6 Additional settings
- API key editing
- Black/white theme switch
- Default project root path editing

## 3. Technical Architecture

### 3.1 Desktop stack
- Electron (main process + renderer)
- React + Vite in renderer
- Electron IPC between renderer and main process

### 3.2 Backend integration strategy
- Use Python backend (`backend_core.py`) as execution engine.
- Main process invokes backend commands via one of:
  - Option A: child process execution
  - Option B: local HTTP service wrapper
- Recommended first implementation: Option A (simpler packaging path).

### 3.3 Configuration and storage
- Global app config file (API key, theme, last project path)
- Project-local files for workflow data and outputs

## 4. Project Folder Structure Specification

```text
<project_root>/
  project.json
  stage1/
    ocr_input/...
    table.csv
    ocr_result.json
  stage2/
    analysis_input.csv
    analysis_result.json
    plots/
      *.png
  stage3/
    uncertainty_input.json
    uncertainty_result.json
  stage4/
    chat_history.json
    draft.md
    draft.tex
```

## 5. Functional Modules

### 5.1 App shell and navigation
- Left navigation or top stepper for 4 stages
- Current stage status badges (pending / done)
- Global project selector and settings access

### 5.2 Stage 1: Table recognition
- Import image(s)
- Call backend OCR API (`extract_table_from_image_bytes`)
- Preview/edit table
- Save to `stage1/table.csv`
- One-click push to Stage 2 input

### 5.3 Stage 2: Analysis and plotting
- Read table data from Stage 1 or manual upload
- Select fitting method and parameters
- Call backend `DataAnalyzer` methods
- Render plot preview and metrics
- Save JSON + PNG outputs under `stage2`
- One-click push to Stage 4 context

### 5.4 Stage 3: Uncertainty analysis
- Input formula and measurements
- Call backend `UncertaintyCalculator` and/or LLM-assisted flow
- Display total uncertainty, relative uncertainty, contributions
- Save outputs under `stage3`
- One-click push to Stage 4 context

### 5.5 Stage 4: AI chat and writing
- Load context from Stage 1/2/3 or manual attachments
- Multi-turn chat with persisted history
- Generate report fragments
- Save `chat_history.json`, `draft.md` and optional `draft.tex`

### 5.6 Settings
- API key view/edit
- Theme toggle (black/white)
- Default project root path view/edit
- Optional: export/import app config

### 5.7 Visual style baseline (minimal geek)
- Minimal monochrome design language (black/white/gray)
- One accent color only for actionable state
- Low-radius, low-shadow, clean grid layout
- Monospace-priority typography for data and status
- No heavy animation, only subtle transitions

## 6. Interface Contract (Desktop -> Backend)

### 6.1 OCR
- `ocr.extractTableFromImageBytes(imageBytes, mimeType) -> tableData`

### 6.2 Analysis
- `analysis.linearFit(payload)`
- `analysis.logFit(payload)`
- `analysis.powerFit(payload)`
- `analysis.fft(payload)`
- `analysis.plot(payload) -> pngBase64`

### 6.3 Uncertainty
- `uncertainty.compute(payload) -> uncertaintyResult`
- `uncertainty.validate(payload) -> validationResult`

### 6.4 LLM
- `llm.generateText(payload) -> text`
- `llm.streamText(payload) -> stream`
- `llm.computeUncertaintyWithLLM(payload) -> result`

## 7. Packaging and Distribution (.exe)

### 7.1 Build targets
- Electron renderer build
- Electron main process build
- Bundle backend python runtime and `backend_core.py`

### 7.2 Tooling
- `electron-builder` for Windows `nsis` target
- Release output: installer `.exe` and portable build (optional)

### 7.3 Runtime checks
- Validate writable config directory
- Validate project folder permissions
- Validate API key before first LLM call

## 8. Milestones

### M1: Foundation
- Electron + React skeleton
- Config persistence (API key + theme)
- Project selection and naming

### M2: Workflow UI
- Four-stage pages with local file persistence
- Stage status and one-click handoff

### M3: Backend bridge
- IPC + Python invocation
- OCR/analysis/uncertainty wired end-to-end

### M4: AI writing
- Chat history persistence
- Draft generation and export

### M5: Release
- Windows exe packaging
- Smoke testing on clean machine

## 9. Risks and Mitigations
- Python/Electron process communication complexity:
  - Mitigate by defining strict JSON I/O contracts early.
- Packaging size growth:
  - Mitigate by trimming dependencies and excluding dev assets.
- Path and permission issues on Windows:
  - Mitigate with startup checks and user-friendly error prompts.
- API key security:
  - Mitigate with OS user-scope config and optional key obfuscation.

## 10. Acceptance Criteria
- User can complete all 4 stages from one desktop app session.
- Artifacts are fully saved inside selected project folder.
- API key is persisted and editable.
- Theme toggles between black/white correctly.
- Installer `.exe` can be used on a clean Windows machine.
