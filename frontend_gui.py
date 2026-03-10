import base64
import hashlib
import io
import json
import os
import re
import threading
import tkinter as tk
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk
import pandas as pd
from PIL import Image
from tkinter import filedialog, messagebox, ttk

from backend_core import DataAnalyzer, LLMProcessor, UncertaintyCalculator, validate_measurement_data


APP_DIR = Path.home() / ".physics_lab_report_assistant"
CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_CONFIG = {
    "api_key": "",
    "theme": "dark",
    "default_project_root": str(Path.home()),
    "professional_mode": False,
    "font_size": 15,
    "selected_model": "qwen3.5-flash",
    "wheel_speed": 8,
}

MODEL_OPTIONS = ["qwen3.5-flash", "qwen3.5-plus", "qwen3-max"]

TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".tex",
    ".py",
    ".yaml",
    ".yml",
    ".ini",
    ".toml",
    ".log",
    ".xml",
    ".html",
    ".css",
    ".js",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_STAGE4_CONTEXT_FILES = 24
MAX_STAGE4_CONTEXT_CHARS = 20000
MAX_STAGE4_FILE_CHARS = 1600


class PhysicsLabApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Physics Lab Report Assistant")
        self.geometry("660x430")
        self.minsize(550, 380)

        self.config_data = self._load_config()
        ctk.set_appearance_mode(self.config_data.get("theme", "dark"))
        ctk.set_default_color_theme("blue")
        ctk.set_widget_scaling(self._font_size() / 15.0)
        self._init_treeview_styles()

        self.current_project_root: Optional[Path] = None
        self.current_project_name: Optional[str] = None
        self.current_image_path: Optional[Path] = None
        self.current_analysis_image_ctk: Optional[ctk.CTkImage] = None
        self.stage1_df: Optional[pd.DataFrame] = None
        self.stage2_loaded_df: Optional[pd.DataFrame] = None
        self.chat_history: List[Dict[str, str]] = []
        self.stage4_pending_report: str = ""
        self.stage4_streaming_label: Optional[ctk.CTkLabel] = None
        self.stage4_file_items: List[Dict[str, Any]] = []
        self.stage4_file_vars: Dict[str, tk.BooleanVar] = {}
        self.stage4_has_asked_questions = False
        self.stage4_llm_busy = False

        self.analyzer = DataAnalyzer()

        self._build_layout()
        self._bind_global_fast_mousewheel()
        self._refresh_project_status()

        self.after(300, self._run_startup_requirements)

    def _load_config(self) -> Dict[str, Any]:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = DEFAULT_CONFIG.copy()
        else:
            data = DEFAULT_CONFIG.copy()

        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        self._save_config(merged)
        return merged

    def _save_config(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        data = cfg if cfg is not None else self.config_data
        APP_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _show_first_launch_hint(self) -> None:
        messagebox.showinfo(
            "First Launch",
            "Please set DASHSCOPE API Key in Settings before using OCR/LLM features.",
        )

    def _init_treeview_styles(self) -> None:
        base_size = self._font_size()
        style = ttk.Style(self)
        style.configure("App.Treeview", font=("Segoe UI", max(11, base_size - 2)), rowheight=30)
        style.configure("App.Treeview.Heading", font=("Segoe UI", max(11, base_size - 2), "bold"))
        style.configure("Stage3.Treeview", font=("Segoe UI", max(12, base_size - 1)), rowheight=34)
        style.configure("Stage3.Treeview.Heading", font=("Segoe UI", max(12, base_size - 1), "bold"))

    def _font_size(self) -> int:
        raw = self.config_data.get("font_size", 15)
        try:
            size = int(raw)
        except Exception:
            size = 15
        return min(24, max(12, size))

    def _selected_model(self) -> str:
        model = str(self.config_data.get("selected_model", "qwen3.5-flash") or "qwen3.5-flash")
        return model if model in MODEL_OPTIONS else "qwen3.5-flash"

    def _selected_enable_thinking(self) -> bool:
        model = self._selected_model().lower()
        return model in {"qwen3.5-plus", "qwen3-max"}

    def _wheel_speed(self) -> int:
        raw = self.config_data.get("wheel_speed", 8)
        try:
            speed = int(raw)
        except Exception:
            speed = 8
        return min(20, max(1, speed))

    def _apply_text_fonts(self) -> None:
        ctk.set_widget_scaling(self._font_size() / 15.0)
        size = self._font_size()
        widgets = [
            "stage2_result_box",
            "stage3_result_box",
            "stage4_context_box",
            "stage4_prompt_entry",
        ]
        for name in widgets:
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.configure(font=ctk.CTkFont(size=size))
                except Exception:
                    pass

        if hasattr(self, "stage4_chat_frame"):
            for row in self.stage4_chat_frame.winfo_children():
                for child in row.winfo_children():
                    if isinstance(child, ctk.CTkLabel):
                        try:
                            child.configure(font=ctk.CTkFont(size=size))
                        except Exception:
                            pass

    def _run_startup_requirements(self) -> None:
        if not self._ensure_api_key_required():
            return
        if not self._ensure_folder_selected_required():
            return

    def _ensure_api_key_required(self) -> bool:
        while not self.config_data.get("api_key", "").strip():
            messagebox.showwarning(
                "API Key Required",
                f"请先填写 DASHSCOPE API Key。\n配置文件位置：{CONFIG_PATH}",
            )
            win = self.open_settings_window(force_api_key=True)
            self.wait_window(win)
            if self.config_data.get("api_key", "").strip():
                break

            retry = messagebox.askretrycancel(
                "API Key Required",
                "未检测到 API Key。点击 Retry 继续填写，点击 Cancel 退出程序。",
            )
            if not retry:
                self.destroy()
                return False
        return True

    def _ensure_folder_selected_required(self) -> bool:
        while self.current_project_root is None:
            messagebox.showinfo("Open Folder", "请先选择一个工作文件夹。")
            self.select_project_root()
            if self.current_project_root is not None:
                break

            retry = messagebox.askretrycancel(
                "Folder Required",
                "必须先打开文件夹才能继续。点击 Retry 重新选择，点击 Cancel 退出程序。",
            )
            if not retry:
                self.destroy()
                return False
        return True

    def _build_layout(self) -> None:
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar_visible = True
        self.sidebar_width = 300

        self.sidebar_container = ctk.CTkFrame(self, corner_radius=0, width=300)
        self.sidebar_container.grid(row=0, column=0, sticky="nsew")
        self.sidebar_container.grid_propagate(False)
        self.grid_columnconfigure(0, minsize=self.sidebar_width, weight=0)

        self.sidebar_resizer = ctk.CTkFrame(self, width=6, corner_radius=0, fg_color=("#BFC4CC", "#3A3F47"))
        self.sidebar_resizer.grid(row=0, column=1, sticky="ns")
        self.sidebar_resizer.bind("<B1-Motion>", self._on_sidebar_resizer_drag)
        self.sidebar_resizer.bind("<Button-1>", lambda _e: self.sidebar_resizer.configure(cursor="sb_h_double_arrow"))
        self.sidebar_resizer.bind("<ButtonRelease-1>", lambda _e: self.sidebar_resizer.configure(cursor="arrow"))

        self.sidebar = ctk.CTkScrollableFrame(self.sidebar_container, corner_radius=0)
        self.sidebar.pack(fill="both", expand=True)
        self.sidebar.grid_rowconfigure(16, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text="Physics Lab Assistant",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(20, 10), sticky="w")

        ctk.CTkButton(self.sidebar, text="Select Project Folder", command=self.select_project_root).grid(
            row=1, column=0, padx=16, pady=6, sticky="ew"
        )
        ctk.CTkButton(self.sidebar, text="Open Project Folder", command=self.open_project_folder).grid(
            row=2, column=0, padx=16, pady=6, sticky="ew"
        )
        ctk.CTkLabel(self.sidebar, text="Project Name").grid(row=3, column=0, padx=16, pady=(10, 2), sticky="w")
        self.project_name_entry = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. optics_exp_01")
        self.project_name_entry.grid(row=4, column=0, padx=16, pady=4, sticky="ew")
        ctk.CTkButton(self.sidebar, text="Create / Open Project", command=self.create_or_open_project).grid(
            row=5, column=0, padx=16, pady=8, sticky="ew"
        )

        self.project_root_label = ctk.CTkLabel(self.sidebar, text="Root: (not selected)", wraplength=260, justify="left")
        self.project_root_label.grid(row=6, column=0, padx=16, pady=(6, 2), sticky="w")
        self.project_path_label = ctk.CTkLabel(self.sidebar, text="Project: (not open)", wraplength=260, justify="left")
        self.project_path_label.grid(row=7, column=0, padx=16, pady=(2, 10), sticky="w")

        self.stage_status_label = ctk.CTkLabel(self.sidebar, text="Stage status\nS1: pending\nS2: pending\nS3: pending\nS4: pending", justify="left")
        self.stage_status_label.grid(row=8, column=0, padx=16, pady=8, sticky="w")

        self.workflow_progress = ctk.CTkProgressBar(self.sidebar)
        self.workflow_progress.grid(row=9, column=0, padx=16, pady=(4, 2), sticky="ew")
        self.workflow_progress.set(0)
        self.task_status_label = ctk.CTkLabel(self.sidebar, text="Status: Idle", wraplength=260, justify="left")
        self.task_status_label.grid(row=10, column=0, padx=16, pady=(0, 8), sticky="w")

        ctk.CTkLabel(self.sidebar, text="Theme").grid(row=11, column=0, padx=16, pady=(16, 4), sticky="w")
        self.theme_option = ctk.CTkOptionMenu(self.sidebar, values=["dark", "light"], command=self.change_theme)
        self.theme_option.set(self.config_data.get("theme", "dark"))
        self.theme_option.grid(row=12, column=0, padx=16, pady=4, sticky="ew")

        ctk.CTkButton(self.sidebar, text="Settings", command=self.open_settings_window).grid(
            row=13, column=0, padx=16, pady=(12, 6), sticky="ew"
        )
        ctk.CTkButton(self.sidebar, text="Save All", command=self.save_all).grid(
            row=14, column=0, padx=16, pady=6, sticky="ew"
        )

        self.main_panel = ctk.CTkFrame(self)
        self.main_panel.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self.main_panel.grid_columnconfigure(0, weight=1)
        self.main_panel.grid_rowconfigure(1, weight=1)

        header_bar = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        header_bar.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        header_bar.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(header_bar, text="Sidebar", width=90, command=self.toggle_sidebar).grid(
            row=0, column=0, padx=(0, 8), sticky="w"
        )

        self.header_label = ctk.CTkLabel(
            header_bar,
            text="Four-Stage Workflow",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.header_label.grid(row=0, column=1, sticky="w")

        ctk.CTkButton(header_bar, text="< Prev", width=96, command=self._go_prev_tab).grid(
            row=0, column=2, padx=(8, 4), sticky="e"
        )
        ctk.CTkButton(header_bar, text="Next >", width=96, command=self._go_next_tab).grid(
            row=0, column=3, padx=(4, 0), sticky="e"
        )

        self.tabview = ctk.CTkTabview(self.main_panel)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))

        self.tab_stage1_container = self.tabview.add("Stage 1 - OCR")
        self.tab_stage2_container = self.tabview.add("Stage 2 - Analysis")
        self.tab_stage3_container = self.tabview.add("Stage 3 - Uncertainty")
        self.tab_stage4_container = self.tabview.add("Stage 4 - AI Writing")

        self.tab_stage1 = ctk.CTkScrollableFrame(self.tab_stage1_container)
        self.tab_stage1.pack(fill="both", expand=True)
        self.tab_stage2 = ctk.CTkScrollableFrame(self.tab_stage2_container)
        self.tab_stage2.pack(fill="both", expand=True)
        self.tab_stage3 = ctk.CTkScrollableFrame(self.tab_stage3_container)
        self.tab_stage3.pack(fill="both", expand=True)
        self.tab_stage4 = ctk.CTkScrollableFrame(self.tab_stage4_container)
        self.tab_stage4.pack(fill="both", expand=True)

        self._build_stage1()
        self._build_stage2()
        self._build_stage3()
        self._build_stage4()
        self._apply_text_fonts()

    def toggle_sidebar(self) -> None:
        if self.sidebar_visible:
            self.sidebar_container.grid_remove()
            self.sidebar_resizer.grid_remove()
            self.sidebar_visible = False
        else:
            self.sidebar_container.grid()
            self.sidebar_resizer.grid()
            self.sidebar_visible = True

    def _on_sidebar_resizer_drag(self, event: tk.Event) -> None:
        if not self.sidebar_visible:
            return
        new_width = int(event.x_root - self.winfo_rootx())
        new_width = max(200, min(520, new_width))
        self.sidebar_width = new_width
        self.grid_columnconfigure(0, minsize=self.sidebar_width)
        self.sidebar_container.configure(width=self.sidebar_width)

    def _scroll_sidebar(self, units: int) -> None:
        canvas = getattr(self.sidebar, "_parent_canvas", None)
        if canvas is None:
            return
        try:
            canvas.yview_scroll(units, "units")
        except Exception:
            pass

    def _set_task_status(self, text: str) -> None:
        if hasattr(self, "task_status_label"):
            self.task_status_label.configure(text=f"Status: {text}")
            self.update_idletasks()

    def _sanitize_markdown_asterisk(self, text: str) -> str:
        """Remove markdown '*' markers for chat display while keeping math/operator stars like a*b."""
        cleaned = text.replace("\r\n", "\n")
        # Bullet markers: "* item" -> "item"
        cleaned = re.sub(r"(?m)^\s*\*\s+", "", cleaned)
        # Bold/italic wrappers: **text** / *text* -> text
        cleaned = re.sub(r"(?<![\w\d])\*\*([^*\n]+)\*\*(?![\w\d])", r"\1", cleaned)
        cleaned = re.sub(r"(?<![\w\d])\*([^*\n]+)\*(?![\w\d])", r"\1", cleaned)
        return cleaned

    def _bind_global_fast_mousewheel(self) -> None:
        self.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")
        self.bind_all("<Control-MouseWheel>", self._on_ctrl_mousewheel, add="+")
        sidebar_canvas = getattr(self.sidebar, "_parent_canvas", None)
        if sidebar_canvas is not None:
            sidebar_canvas.bind("<MouseWheel>", self._on_sidebar_mousewheel, add="+")

    def _on_global_mousewheel(self, event: tk.Event) -> None:
        if int(getattr(event, "state", 0)) & 0x4:
            return

        current = self.tabview.get()
        frame_map = {
            "Stage 1 - OCR": self.tab_stage1,
            "Stage 2 - Analysis": self.tab_stage2,
            "Stage 3 - Uncertainty": self.tab_stage3,
            "Stage 4 - AI Writing": self.tab_stage4,
        }
        frame = frame_map.get(current)
        if frame is None:
            return

        canvas = getattr(frame, "_parent_canvas", None)
        if canvas is None:
            return

        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return

        # Increase wheel speed so each notch scrolls farther.
        steps = int(-delta / 120)
        if steps == 0:
            steps = -1 if delta > 0 else 1
        canvas.yview_scroll(steps * self._wheel_speed(), "units")

    def _on_ctrl_mousewheel(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        current = self._font_size()
        if delta > 0:
            new_size = min(24, current + 1)
        else:
            new_size = max(12, current - 1)
        if new_size == current:
            return

        self.config_data["font_size"] = new_size
        self._save_config()
        self._init_treeview_styles()
        self._apply_text_fonts()

    def _on_sidebar_mousewheel(self, event: tk.Event) -> None:
        if int(getattr(event, "state", 0)) & 0x4:
            return
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        steps = int(-delta / 120)
        if steps == 0:
            steps = -1 if delta > 0 else 1
        self._scroll_sidebar(steps * self._wheel_speed())

    def _set_treeview_dataframe(self, tree: ttk.Treeview, df: pd.DataFrame) -> None:
        cols = [str(c) for c in df.columns]
        if not cols:
            cols = ["(empty)"]
            df = pd.DataFrame(columns=cols)

        tree.delete(*tree.get_children())
        tree.configure(columns=cols, show="headings")

        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor="center", stretch=True)

        for _, row in df.iterrows():
            values: List[str] = []
            for col in cols:
                val = row[col] if col in df.columns else ""
                values.append("" if pd.isna(val) else str(val))
            tree.insert("", "end", values=values)

        if hasattr(self, "stage1_table_tree") and tree is self.stage1_table_tree and hasattr(self, "stage1_header_entry"):
            self.stage1_header_entry.delete(0, "end")
            self.stage1_header_entry.insert(0, ", ".join(cols))

    def _treeview_to_dataframe(self, tree: ttk.Treeview) -> pd.DataFrame:
        columns = [str(c) for c in tree.cget("columns")]
        if not columns:
            return pd.DataFrame()

        rows: List[List[Any]] = []
        for item_id in tree.get_children():
            values = list(tree.item(item_id, "values"))
            if len(values) < len(columns):
                values.extend([""] * (len(columns) - len(values)))
            rows.append(values[: len(columns)])
        return pd.DataFrame(rows, columns=columns)

    def _enable_treeview_cell_edit(self, tree: ttk.Treeview) -> None:
        tree.bind("<Double-1>", lambda e: self._edit_treeview_cell(tree, e), add="+")

    def _edit_treeview_cell(self, tree: ttk.Treeview, event: tk.Event) -> None:
        if tree.identify("region", event.x, event.y) != "cell":
            return

        row_id = tree.identify_row(event.y)
        col_id = tree.identify_column(event.x)
        if not row_id or not col_id:
            return

        bbox = tree.bbox(row_id, col_id)
        if not bbox:
            return

        x, y, width, height = bbox
        col_idx = int(col_id[1:]) - 1
        values = list(tree.item(row_id, "values"))
        old_val = values[col_idx] if col_idx < len(values) else ""

        editor = tk.Entry(tree)
        editor.place(x=x, y=y, width=width, height=height)
        editor.insert(0, old_val)
        editor.focus_set()

        def commit_edit(_event: Optional[tk.Event] = None) -> None:
            new_val = editor.get()
            current = list(tree.item(row_id, "values"))
            if len(current) < len(tree.cget("columns")):
                current.extend([""] * (len(tree.cget("columns")) - len(current)))
            current[col_idx] = new_val
            tree.item(row_id, values=current)
            editor.destroy()

        def cancel_edit(_event: Optional[tk.Event] = None) -> None:
            if editor.winfo_exists():
                editor.destroy()

        editor.bind("<Return>", commit_edit)
        editor.bind("<FocusOut>", commit_edit)
        editor.bind("<Escape>", cancel_edit)

    def _apply_stage1_headers(self) -> None:
        df = self._treeview_to_dataframe(self.stage1_table_tree)
        if df.empty:
            messagebox.showwarning("No Table", "Please load OCR data first.")
            return

        raw = self.stage1_header_entry.get().strip()
        if not raw:
            messagebox.showwarning("Missing Headers", "Please input header names.")
            return

        headers = [h.strip() for h in raw.split(",")]
        if len(headers) != len(df.columns):
            messagebox.showwarning("Header Count Mismatch", f"Need {len(df.columns)} headers.")
            return

        unique_headers: List[str] = []
        for idx, name in enumerate(headers):
            base = name or f"col_{idx+1}"
            unique_headers.append(self._make_unique_name(base, unique_headers))

        df.columns = unique_headers
        self._set_treeview_dataframe(self.stage1_table_tree, df)

    def _use_first_row_as_header(self) -> None:
        df = self._treeview_to_dataframe(self.stage1_table_tree)
        if df.empty:
            messagebox.showwarning("No Table", "Please load OCR data first.")
            return
        if len(df.index) < 1:
            messagebox.showwarning("No Data", "Table has no rows.")
            return

        first_row = [str(v).strip() for v in df.iloc[0].tolist()]
        new_headers: List[str] = []
        for idx, val in enumerate(first_row):
            base = val or f"col_{idx+1}"
            new_headers.append(self._make_unique_name(base, new_headers))

        body = df.iloc[1:].reset_index(drop=True)
        body.columns = new_headers
        self._set_treeview_dataframe(self.stage1_table_tree, body)

    def _make_unique_name(self, base: str, existing: List[str]) -> str:
        name = base
        i = 2
        while name in existing:
            name = f"{base}_{i}"
            i += 1
        return name

    def _auto_axis_label(self, ref: str, mode: str, axis: str) -> str:
        base = ref.strip() if ref.strip() else f"{axis}"
        if mode == "row":
            return f"row_{base}"
        return base

    def _sanitize_file_stem(self, raw: str, default: str) -> str:
        txt = (raw or "").strip()
        if not txt:
            txt = default
        allowed = []
        for ch in txt:
            if ch.isalnum() or ch in {"_", "-"}:
                allowed.append(ch)
            else:
                allowed.append("_")
        stem = "".join(allowed).strip("_")
        return stem or default

    def _unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        i = 2
        while True:
            candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
            if not candidate.exists():
                return candidate
            i += 1

    def _find_first_file_by_suffix(self, folder: Path, suffixes: set[str], name_contains: Optional[str] = None) -> Optional[Path]:
        if not folder.exists():
            return None
        files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes])
        if name_contains:
            key = name_contains.lower()
            filtered = [p for p in files if key in p.name.lower()]
            if filtered:
                return filtered[0]
        return files[0] if files else None

    def _default_stage1_table_stem(self, df: pd.DataFrame) -> str:
        cols = [self._sanitize_file_stem(str(c), "col") for c in df.columns[:2]]
        if cols:
            return self._sanitize_file_stem("table_" + "_".join(cols), "table_data")
        return "table_data"

    def _default_stage2_plot_stem(self, result: Dict[str, Any]) -> str:
        method = self._sanitize_file_stem(str(result.get("method", "plot")), "plot")
        x_ref = self._sanitize_file_stem(str(result.get("x_ref", "x")), "x")
        y_ref = self._sanitize_file_stem(str(result.get("y_ref", "y")), "y")
        return f"plot_{method}_{y_ref}_vs_{x_ref}"

    def _build_stage1(self) -> None:
        self.tab_stage1.grid_columnconfigure(0, weight=1)
        self.tab_stage1.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(self.tab_stage1, text="Choose Image", command=self.select_ocr_image).grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )
        ctk.CTkButton(self.tab_stage1, text="Run OCR", command=self.run_ocr).grid(
            row=0, column=1, padx=10, pady=10, sticky="e"
        )

        self.stage1_image_label = ctk.CTkLabel(self.tab_stage1, text="Image: (none)", anchor="w")
        self.stage1_image_label.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(
            self.tab_stage1,
            text="OCR result is editable below. Modify CSV content before saving if needed.",
            anchor="w",
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 4), sticky="ew")

        header_row = ctk.CTkFrame(self.tab_stage1, fg_color="transparent")
        header_row.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 6), sticky="ew")
        header_row.grid_columnconfigure(0, weight=1)
        self.stage1_header_entry = ctk.CTkEntry(header_row, placeholder_text="Edit headers as comma-separated names")
        self.stage1_header_entry.grid(row=0, column=0, padx=(0, 6), pady=2, sticky="ew")
        ctk.CTkButton(header_row, text="Apply Headers", width=120, command=self._apply_stage1_headers).grid(
            row=0, column=1, padx=4, pady=2
        )
        ctk.CTkButton(header_row, text="Use First Row As Header", width=170, command=self._use_first_row_as_header).grid(
            row=0, column=2, padx=4, pady=2
        )

        name_row = ctk.CTkFrame(self.tab_stage1, fg_color="transparent")
        name_row.grid(row=4, column=0, columnspan=2, padx=10, pady=(0, 6), sticky="ew")
        name_row.grid_columnconfigure(1, weight=1)
        name_row.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(name_row, text="Table Save Name").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.stage1_table_name_entry = ctk.CTkEntry(name_row, placeholder_text="auto")
        self.stage1_table_name_entry.grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        ctk.CTkLabel(name_row, text="Image Save Name").grid(row=0, column=2, padx=4, pady=2, sticky="w")
        self.stage1_image_name_entry = ctk.CTkEntry(name_row, placeholder_text="auto")
        self.stage1_image_name_entry.grid(row=0, column=3, padx=4, pady=2, sticky="ew")

        self.stage1_table_frame = ctk.CTkFrame(self.tab_stage1)
        self.stage1_table_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=8, sticky="nsew")
        self.stage1_table_frame.grid_columnconfigure(0, weight=1)
        self.stage1_table_frame.grid_rowconfigure(0, weight=1)

        self.stage1_table_tree = ttk.Treeview(self.stage1_table_frame, show="headings", height=12, style="App.Treeview")
        self.stage1_table_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(self.stage1_table_frame, orient="vertical", command=self.stage1_table_tree.yview)
        x_scroll = ttk.Scrollbar(self.stage1_table_frame, orient="horizontal", command=self.stage1_table_tree.xview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.stage1_table_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self._enable_treeview_cell_edit(self.stage1_table_tree)

        btn_row = ctk.CTkFrame(self.tab_stage1, fg_color="transparent")
        btn_row.grid(row=6, column=0, columnspan=2, padx=10, pady=8, sticky="ew")
        btn_row.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(btn_row, text="Save Stage 1", command=self.save_stage1).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(btn_row, text="Push to Stage 2", command=self.push_stage1_to_stage2).grid(
            row=0, column=1, padx=4, pady=4, sticky="ew"
        )

    def _build_stage2(self) -> None:
        self.tab_stage2.grid_columnconfigure(0, weight=1)
        self.tab_stage2.grid_columnconfigure(1, weight=1)

        input_frame = ctk.CTkFrame(self.tab_stage2)
        input_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        input_frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(input_frame, text="CSV Path").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.stage2_csv_path = ctk.CTkEntry(input_frame, placeholder_text="Select a csv file")
        self.stage2_csv_path.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(input_frame, text="Browse", width=90, command=self.select_stage2_csv).grid(
            row=0, column=2, padx=6, pady=6
        )
        ctk.CTkButton(input_frame, text="Load Table", width=90, command=self.load_stage2_table).grid(
            row=0, column=3, padx=6, pady=6
        )

        ctk.CTkLabel(input_frame, text="Method").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.stage2_method = ctk.CTkOptionMenu(input_frame, values=["linear", "log", "power", "fft"])
        self.stage2_method.set("linear")
        self.stage2_method.grid(row=1, column=1, padx=6, pady=6, sticky="w")

        ctk.CTkLabel(input_frame, text="Extract By").grid(row=1, column=2, padx=6, pady=6, sticky="w")
        self.stage2_extract_mode = ctk.CTkOptionMenu(
            input_frame,
            values=["column", "row"],
            command=lambda _v: self._refresh_stage2_selection_menus(),
        )
        self.stage2_extract_mode.set("column")
        self.stage2_extract_mode.grid(row=1, column=3, padx=6, pady=6, sticky="w")

        ctk.CTkLabel(input_frame, text="X Selection").grid(row=2, column=0, padx=6, pady=6, sticky="w")
        self.stage2_x_select = ctk.CTkOptionMenu(input_frame, values=["(none)"])
        self.stage2_x_select.set("(none)")
        self.stage2_x_select.grid(row=2, column=1, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(input_frame, text="Y Selection").grid(row=3, column=0, padx=6, pady=6, sticky="w")
        self.stage2_y_select = ctk.CTkOptionMenu(input_frame, values=["(none)"])
        self.stage2_y_select.set("(none)")
        self.stage2_y_select.grid(row=3, column=1, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(input_frame, text="Y Err Selection (optional)").grid(row=4, column=0, padx=6, pady=6, sticky="w")
        self.stage2_yerr_select = ctk.CTkOptionMenu(input_frame, values=["(none)"])
        self.stage2_yerr_select.set("(none)")
        self.stage2_yerr_select.grid(row=4, column=1, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(input_frame, text="Data Range (start:end)").grid(row=2, column=2, padx=6, pady=6, sticky="w")
        range_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        range_frame.grid(row=2, column=3, padx=6, pady=6, sticky="ew")
        self.stage2_start_index = ctk.CTkEntry(range_frame, width=70, placeholder_text="start")
        self.stage2_start_index.grid(row=0, column=0, padx=(0, 6), pady=2)
        self.stage2_end_index = ctk.CTkEntry(range_frame, width=70, placeholder_text="end")
        self.stage2_end_index.grid(row=0, column=1, padx=(0, 6), pady=2)
        ctk.CTkLabel(range_frame, text="(blank = all)").grid(row=0, column=2, padx=4, pady=2, sticky="w")

        ctk.CTkLabel(input_frame, text="Plot Title").grid(row=3, column=2, padx=6, pady=6, sticky="w")
        self.stage2_plot_title = ctk.CTkEntry(input_frame, placeholder_text="Custom title")
        self.stage2_plot_title.grid(row=3, column=3, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(input_frame, text="X Axis Label").grid(row=4, column=2, padx=6, pady=6, sticky="w")
        self.stage2_xlabel = ctk.CTkEntry(input_frame, placeholder_text="Custom x label")
        self.stage2_xlabel.grid(row=4, column=3, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(input_frame, text="Y Axis Label").grid(row=5, column=2, padx=6, pady=6, sticky="w")
        self.stage2_ylabel = ctk.CTkEntry(input_frame, placeholder_text="Custom y label")
        self.stage2_ylabel.grid(row=5, column=3, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(input_frame, text="Sampling Rate (FFT)").grid(row=5, column=0, padx=6, pady=6, sticky="w")
        self.stage2_sampling_rate = ctk.CTkEntry(input_frame)
        self.stage2_sampling_rate.insert(0, "1.0")
        self.stage2_sampling_rate.grid(row=5, column=1, padx=6, pady=6, sticky="w")

        ctk.CTkLabel(input_frame, text="Plot Save Name").grid(row=6, column=0, padx=6, pady=6, sticky="w")
        self.stage2_plot_name_entry = ctk.CTkEntry(input_frame, placeholder_text="auto")
        self.stage2_plot_name_entry.grid(row=6, column=1, padx=6, pady=6, sticky="ew")

        ctk.CTkButton(self.tab_stage2, text="Run Analysis", command=self.run_stage2_analysis).grid(
            row=1, column=0, padx=10, pady=6, sticky="w"
        )
        ctk.CTkButton(self.tab_stage2, text="Push to Stage 4", command=self.push_stage2_to_stage4).grid(
            row=1, column=1, padx=10, pady=6, sticky="e"
        )

        ctk.CTkLabel(self.tab_stage2, text="Loaded Table Preview", anchor="w").grid(
            row=2, column=0, columnspan=2, padx=10, pady=(4, 2), sticky="ew"
        )

        self.stage2_table_frame = ctk.CTkFrame(self.tab_stage2)
        self.stage2_table_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=6, sticky="nsew")
        self.stage2_table_frame.grid_columnconfigure(0, weight=1)
        self.stage2_table_frame.grid_rowconfigure(0, weight=1)
        self.stage2_table_tree = ttk.Treeview(self.stage2_table_frame, show="headings", height=8, style="App.Treeview")
        self.stage2_table_tree.grid(row=0, column=0, sticky="nsew")
        y2_scroll = ttk.Scrollbar(self.stage2_table_frame, orient="vertical", command=self.stage2_table_tree.yview)
        x2_scroll = ttk.Scrollbar(self.stage2_table_frame, orient="horizontal", command=self.stage2_table_tree.xview)
        y2_scroll.grid(row=0, column=1, sticky="ns")
        x2_scroll.grid(row=1, column=0, sticky="ew")
        self.stage2_table_tree.configure(yscrollcommand=y2_scroll.set, xscrollcommand=x2_scroll.set)

        self.stage2_result_box = ctk.CTkTextbox(self.tab_stage2, height=170)
        self.stage2_result_box.grid(row=4, column=0, columnspan=2, padx=10, pady=8, sticky="ew")
        self.stage2_result_box.configure(font=ctk.CTkFont(size=self._font_size()))

        self.stage2_plot_label = ctk.CTkLabel(self.tab_stage2, text="(plot preview)")
        self.stage2_plot_label.grid(row=5, column=0, columnspan=2, padx=10, pady=8, sticky="ew")

    def _build_stage3(self) -> None:
        self.tab_stage3.grid_columnconfigure(0, weight=1)
        self.tab_stage3.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.tab_stage3, text="Formula").grid(row=0, column=0, padx=10, pady=(12, 4), sticky="w")
        self.stage3_formula = ctk.CTkEntry(self.tab_stage3, placeholder_text="e.g. F = m*a")
        self.stage3_formula.grid(row=0, column=1, padx=10, pady=(12, 4), sticky="ew")

        input_row = ctk.CTkFrame(self.tab_stage3)
        input_row.grid(row=1, column=0, columnspan=2, padx=10, pady=6, sticky="ew")
        input_row.grid_columnconfigure((1, 3, 5, 7, 9), weight=1)

        ctk.CTkLabel(input_row, text="Quantity").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.stage3_qty_entry = ctk.CTkEntry(input_row, placeholder_text="m")
        self.stage3_qty_entry.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        ctk.CTkLabel(input_row, text="Value").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        self.stage3_value_entry = ctk.CTkEntry(input_row, placeholder_text="0.5")
        self.stage3_value_entry.grid(row=0, column=3, padx=4, pady=4, sticky="ew")

        ctk.CTkLabel(input_row, text="Unit").grid(row=0, column=4, padx=4, pady=4, sticky="w")
        self.stage3_unit_entry = ctk.CTkEntry(input_row, placeholder_text="kg")
        self.stage3_unit_entry.grid(row=0, column=5, padx=4, pady=4, sticky="ew")

        ctk.CTkLabel(input_row, text="A Unc.").grid(row=0, column=6, padx=4, pady=4, sticky="w")
        self.stage3_a_entry = ctk.CTkEntry(input_row, placeholder_text="0.001")
        self.stage3_a_entry.grid(row=0, column=7, padx=4, pady=4, sticky="ew")

        ctk.CTkLabel(input_row, text="B Unc.").grid(row=0, column=8, padx=4, pady=4, sticky="w")
        self.stage3_b_entry = ctk.CTkEntry(input_row, placeholder_text="0.0005")
        self.stage3_b_entry.grid(row=0, column=9, padx=4, pady=4, sticky="ew")

        ctk.CTkButton(input_row, text="Add / Update", command=self._stage3_add_or_update_row).grid(
            row=0, column=10, padx=6, pady=4
        )

        self.stage3_table_frame = ctk.CTkFrame(self.tab_stage3)
        self.stage3_table_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=6, sticky="nsew")
        self.stage3_table_frame.grid_columnconfigure(0, weight=1)
        self.stage3_table_frame.grid_rowconfigure(0, weight=1)
        self.stage3_table_tree = ttk.Treeview(self.stage3_table_frame, show="headings", height=9, style="Stage3.Treeview")
        self.stage3_table_tree.grid(row=0, column=0, sticky="nsew")
        y3_scroll = ttk.Scrollbar(self.stage3_table_frame, orient="vertical", command=self.stage3_table_tree.yview)
        x3_scroll = ttk.Scrollbar(self.stage3_table_frame, orient="horizontal", command=self.stage3_table_tree.xview)
        y3_scroll.grid(row=0, column=1, sticky="ns")
        x3_scroll.grid(row=1, column=0, sticky="ew")
        self.stage3_table_tree.configure(yscrollcommand=y3_scroll.set, xscrollcommand=x3_scroll.set)
        self._enable_treeview_cell_edit(self.stage3_table_tree)

        stage3_init_df = pd.DataFrame(columns=["quantity", "value", "unit", "a_uncertainty", "b_uncertainty"])
        self._set_treeview_dataframe(self.stage3_table_tree, stage3_init_df)

        ctk.CTkLabel(
            self.tab_stage3,
            text="Double-click table cells to edit. Required columns: quantity,value,unit,a_uncertainty,b_uncertainty",
            anchor="w",
        ).grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 4), sticky="ew")

        stage3_table_actions = ctk.CTkFrame(self.tab_stage3, fg_color="transparent")
        stage3_table_actions.grid(row=4, column=0, columnspan=2, padx=10, pady=4, sticky="ew")
        stage3_table_actions.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(stage3_table_actions, text="Delete Selected", command=self._stage3_delete_selected_row).grid(
            row=0, column=0, padx=4, pady=4, sticky="ew"
        )
        ctk.CTkButton(stage3_table_actions, text="Clear Table", command=self._stage3_clear_rows).grid(
            row=0, column=1, padx=4, pady=4, sticky="ew"
        )

        ctk.CTkButton(self.tab_stage3, text="Compute Uncertainty", command=self.run_stage3_uncertainty).grid(
            row=5, column=0, padx=10, pady=6, sticky="w"
        )
        ctk.CTkButton(self.tab_stage3, text="Push to Stage 4", command=self.push_stage3_to_stage4).grid(
            row=5, column=1, padx=10, pady=6, sticky="e"
        )

        self.stage3_result_box = ctk.CTkTextbox(self.tab_stage3)
        self.stage3_result_box.grid(row=6, column=0, columnspan=2, padx=10, pady=8, sticky="nsew")
        self.stage3_result_box.configure(font=ctk.CTkFont(size=self._font_size()))

    def _stage3_add_or_update_row(self) -> None:
        qty = self.stage3_qty_entry.get().strip()
        if not qty:
            messagebox.showwarning("Missing Quantity", "Please input quantity symbol.")
            return

        try:
            value = float(self.stage3_value_entry.get().strip() or "0")
            a_unc = float(self.stage3_a_entry.get().strip() or "0")
            b_unc = float(self.stage3_b_entry.get().strip() or "0")
        except ValueError:
            messagebox.showerror("Invalid Number", "Value / A Unc / B Unc must be valid numbers.")
            return

        row = {
            "quantity": qty,
            "value": str(value),
            "unit": self.stage3_unit_entry.get().strip(),
            "a_uncertainty": str(a_unc),
            "b_uncertainty": str(b_unc),
        }
        df = self._treeview_to_dataframe(self.stage3_table_tree)
        if df.empty:
            df = pd.DataFrame(columns=["quantity", "value", "unit", "a_uncertainty", "b_uncertainty"])

        if "quantity" not in df.columns:
            messagebox.showerror("Table Error", "Stage 3 table columns are invalid.")
            return

        mask = df["quantity"].astype(str).str.strip() == qty
        if mask.any():
            idx = df[mask].index[0]
            for k, v in row.items():
                df.at[idx, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        self._set_treeview_dataframe(self.stage3_table_tree, df)
        self.stage3_qty_entry.delete(0, "end")
        self.stage3_value_entry.delete(0, "end")
        self.stage3_unit_entry.delete(0, "end")
        self.stage3_a_entry.delete(0, "end")
        self.stage3_b_entry.delete(0, "end")

    def _stage3_delete_selected_row(self) -> None:
        selected = self.stage3_table_tree.selection()
        if not selected:
            messagebox.showinfo("No Selection", "Please select at least one row to delete.")
            return
        for item in selected:
            self.stage3_table_tree.delete(item)

    def _stage3_clear_rows(self) -> None:
        if not messagebox.askyesno("Confirm", "Clear all measurement rows?"):
            return
        self.stage3_table_tree.delete(*self.stage3_table_tree.get_children())

    def _build_stage4(self) -> None:
        self.tab_stage4.grid_columnconfigure(0, weight=1)
        self.tab_stage4.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.tab_stage4, text="Context (auto: folder+stages, editable for manual notes)").grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        self.stage4_context_box = ctk.CTkTextbox(self.tab_stage4, height=120)
        self.stage4_context_box.grid(row=1, column=0, columnspan=2, padx=10, pady=6, sticky="ew")
        self.stage4_context_box.configure(font=ctk.CTkFont(size=self._font_size()))

        ctk.CTkLabel(self.tab_stage4, text="Scanned Files By Type (deduplicated, default all selected)").grid(
            row=2, column=0, padx=10, pady=(4, 2), sticky="w"
        )

        file_action_row = ctk.CTkFrame(self.tab_stage4, fg_color="transparent")
        file_action_row.grid(row=2, column=1, padx=10, pady=(4, 2), sticky="e")
        ctk.CTkButton(file_action_row, text="Select All", width=96, command=self._stage4_select_all_files).grid(
            row=0, column=0, padx=4, pady=2
        )
        ctk.CTkButton(file_action_row, text="Select None", width=96, command=self._stage4_clear_all_files).grid(
            row=0, column=1, padx=4, pady=2
        )

        self.stage4_file_select_frame = ctk.CTkScrollableFrame(self.tab_stage4, height=140)
        self.stage4_file_select_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=6, sticky="ew")
        self.stage4_file_select_frame.grid_columnconfigure(0, weight=1)

        top_actions = ctk.CTkFrame(self.tab_stage4, fg_color="transparent")
        top_actions.grid(row=0, column=1, padx=10, pady=(8, 4), sticky="e")
        self.stage4_btn_load = ctk.CTkButton(
            top_actions,
            text="Load Folder + Stage Data",
            command=self.load_all_stage_data_to_context,
        )
        self.stage4_btn_load.grid(row=0, column=0, padx=4, pady=4)
        self.stage4_btn_ask = ctk.CTkButton(
            top_actions,
            text="AI Understand + Ask",
            command=self.stage4_ai_ask_questions,
        )
        self.stage4_btn_ask.grid(row=0, column=1, padx=4, pady=4)

        self.stage4_chat_frame = ctk.CTkScrollableFrame(self.tab_stage4)
        self.stage4_chat_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=6, sticky="nsew")
        self.stage4_chat_frame.grid_columnconfigure(0, weight=1)

        self.stage4_prompt_entry = ctk.CTkEntry(self.tab_stage4, placeholder_text="Ask AI to draft/report/interpret result...")
        self.stage4_prompt_entry.grid(row=5, column=0, padx=10, pady=8, sticky="ew")
        self.stage4_prompt_entry.configure(font=ctk.CTkFont(size=self._font_size()))
        send_action_row = ctk.CTkFrame(self.tab_stage4, fg_color="transparent")
        send_action_row.grid(row=5, column=1, padx=10, pady=8, sticky="e")
        self.stage4_btn_send = ctk.CTkButton(send_action_row, text="Send", width=92, command=self.send_stage4_message)
        self.stage4_btn_send.grid(
            row=0, column=0, padx=(0, 6), pady=2
        )
        self.stage4_btn_compose = ctk.CTkButton(
            send_action_row,
            text="Compose Report TeX",
            width=160,
            command=self.generate_stage4_latex_report,
        )
        self.stage4_btn_compose.grid(
            row=0, column=1, padx=(6, 0), pady=2
        )

        action_bar = ctk.CTkFrame(self.tab_stage4, fg_color="transparent")
        action_bar.grid(row=6, column=0, columnspan=2, padx=10, pady=8, sticky="ew")
        action_bar.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(action_bar, text="Save Chat History", command=self.save_stage4_chat).grid(
            row=0, column=0, padx=4, pady=4, sticky="ew"
        )
        ctk.CTkButton(action_bar, text="Save Draft.md", command=self.save_stage4_draft_md).grid(
            row=0, column=1, padx=4, pady=4, sticky="ew"
        )
        ctk.CTkButton(action_bar, text="Save Draft.tex", command=self.save_stage4_draft_tex).grid(
            row=0, column=2, padx=4, pady=4, sticky="ew"
        )

        latex_name_row = ctk.CTkFrame(self.tab_stage4, fg_color="transparent")
        latex_name_row.grid(row=7, column=0, columnspan=2, padx=10, pady=(0, 8), sticky="ew")
        latex_name_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(latex_name_row, text="LaTeX Save Name").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.stage4_tex_name_entry = ctk.CTkEntry(latex_name_row, placeholder_text="auto")
        self.stage4_tex_name_entry.grid(row=0, column=1, padx=4, pady=2, sticky="ew")

    def _append_chat_bubble(self, role: str, content: str) -> ctk.CTkLabel:
        size = self._font_size()
        row_frame = ctk.CTkFrame(self.stage4_chat_frame, fg_color="transparent")
        row_frame.grid(sticky="ew", padx=6, pady=4)
        row_frame.grid_columnconfigure(0, weight=1)

        user_like = role.lower() in {"user"}
        bubble_color = ("#2B6CB0", "#1F4D7A") if user_like else ("#2F855A", "#1E5A3C")
        anchor = "e" if user_like else "w"
        justify = "left"
        label_text = f"{role}:\n{content}" if content else f"{role}:"

        bubble_font = ctk.CTkFont(size=size, family="Consolas") if role.lower().startswith("latex") else ctk.CTkFont(size=size)
        bubble = ctk.CTkLabel(
            row_frame,
            text=label_text,
            justify=justify,
            wraplength=720,
            fg_color=bubble_color,
            corner_radius=12,
            padx=12,
            pady=8,
            font=bubble_font,
        )
        bubble.grid(row=0, column=0, sticky=anchor)

        try:
            self.stage4_chat_frame._parent_canvas.yview_moveto(1.0)  # type: ignore[attr-defined]
        except Exception:
            pass
        return bubble

    def get_project_path(self) -> Optional[Path]:
        if not self.current_project_root or not self.current_project_name:
            return None
        return self.current_project_root / self.current_project_name

    def _tab_names(self) -> List[str]:
        return [
            "Stage 1 - OCR",
            "Stage 2 - Analysis",
            "Stage 3 - Uncertainty",
            "Stage 4 - AI Writing",
        ]

    def _go_prev_tab(self) -> None:
        current = self.tabview.get()
        names = self._tab_names()
        if current not in names:
            self.tabview.set(names[0])
            return
        idx = names.index(current)
        self.tabview.set(names[max(0, idx - 1)])

    def _go_next_tab(self) -> None:
        current = self.tabview.get()
        names = self._tab_names()
        if current not in names:
            self.tabview.set(names[0])
            return
        idx = names.index(current)
        self.tabview.set(names[min(len(names) - 1, idx + 1)])

    def select_project_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.config_data.get("default_project_root", str(Path.home())))
        if not selected:
            return
        self.current_project_root = Path(selected)
        self.config_data["default_project_root"] = selected
        self._save_config()
        self.project_root_label.configure(text=f"Root: {selected}")

    def open_project_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.config_data.get("default_project_root", str(Path.home())))
        if not selected:
            return

        folder = Path(selected)
        self.current_project_root = folder.parent
        self.current_project_name = folder.name
        self.config_data["default_project_root"] = str(folder.parent)
        self._save_config()

        self.project_root_label.configure(text=f"Root: {self.current_project_root}")
        self.project_path_label.configure(text=f"Project: {folder}")
        self.project_name_entry.delete(0, "end")
        self.project_name_entry.insert(0, folder.name)

        # Ensure stage structure exists for newly selected folders.
        (folder / "stage1" / "ocr_input").mkdir(parents=True, exist_ok=True)
        (folder / "stage2" / "plots").mkdir(parents=True, exist_ok=True)
        (folder / "stage3").mkdir(parents=True, exist_ok=True)
        (folder / "stage4").mkdir(parents=True, exist_ok=True)

        self._refresh_project_status()

    def create_or_open_project(self) -> None:
        name = self.project_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Missing Name", "Please input a project name.")
            return
        if not self.current_project_root:
            messagebox.showwarning("Missing Root", "Please select a project root first.")
            return

        self.current_project_name = name
        project_path = self.get_project_path()
        assert project_path is not None

        (project_path / "stage1" / "ocr_input").mkdir(parents=True, exist_ok=True)
        (project_path / "stage2" / "plots").mkdir(parents=True, exist_ok=True)
        (project_path / "stage3").mkdir(parents=True, exist_ok=True)
        (project_path / "stage4").mkdir(parents=True, exist_ok=True)

        project_meta = {
            "project_name": name,
            "project_root": str(project_path),
            "created_by": "frontend_gui.py",
        }
        with open(project_path / "project.json", "w", encoding="utf-8") as f:
            json.dump(project_meta, f, ensure_ascii=False, indent=2)

        self.project_path_label.configure(text=f"Project: {project_path}")
        self._refresh_project_status()
        messagebox.showinfo("Project Ready", f"Project is ready:\n{project_path}")

    def _set_env_api_key(self) -> bool:
        key = self.config_data.get("api_key", "").strip()
        if not key:
            messagebox.showerror("Missing API Key", "Please set API key in Settings.")
            return False
        os.environ["DASHSCOPE_API_KEY"] = key
        return True

    def _new_llm_processor(self) -> Optional[LLMProcessor]:
        if not self._set_env_api_key():
            return None
        try:
            return LLMProcessor(model=self._selected_model())
        except Exception as e:
            messagebox.showerror("LLM Init Error", str(e))
            return None

    def select_ocr_image(self) -> None:
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
            ("All files", "*.*"),
        ]
        selected = filedialog.askopenfilename(filetypes=filetypes)
        if not selected:
            return
        self.current_image_path = Path(selected)
        self.stage1_image_label.configure(text=f"Image: {selected}")

    def run_ocr(self) -> None:
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please choose an image first.")
            return

        processor = self._new_llm_processor()
        if processor is None:
            return

        suffix = self.current_image_path.suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        try:
            self._set_task_status("Running OCR...")
            with open(self.current_image_path, "rb") as f:
                image_bytes = f.read()
            df = processor.extract_table_from_image_bytes(image_bytes, mime_type=mime_type)
            self.stage1_df = df
            self._set_treeview_dataframe(self.stage1_table_tree, df)
            self.tabview.set("Stage 1 - OCR")
        except Exception as e:
            messagebox.showerror("OCR Error", str(e))
        finally:
            self._set_task_status("Idle")

    def save_stage1(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            messagebox.showwarning("No Project", "Please create/open project first.")
            return

        df = self._treeview_to_dataframe(self.stage1_table_tree)
        if df.empty:
            messagebox.showwarning("No Table", "No OCR table content to save.")
            return
        csv_text = df.to_csv(index=False).strip()

        stage1_dir = project_path / "stage1"
        table_stem = self._sanitize_file_stem(self.stage1_table_name_entry.get(), self._default_stage1_table_stem(df))
        table_path = self._unique_path(stage1_dir / f"{table_stem}.csv")
        ocr_json_path = stage1_dir / "ocr_result.json"

        with open(table_path, "w", encoding="utf-8") as f:
            f.write(csv_text + "\n")
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump({"table_csv_path": str(table_path)}, f, ensure_ascii=False, indent=2)

        if self.current_image_path and self.current_image_path.exists():
            image_stem_default = self._sanitize_file_stem(self.current_image_path.stem, "ocr_image")
            image_stem = self._sanitize_file_stem(self.stage1_image_name_entry.get(), image_stem_default)
            copied_path = self._unique_path(stage1_dir / "ocr_input" / f"{image_stem}{self.current_image_path.suffix}")
            if copied_path != self.current_image_path:
                copied_path.write_bytes(self.current_image_path.read_bytes())

        self._refresh_project_status()
        messagebox.showinfo("Saved", f"Stage 1 saved to:\n{table_path}")

    def push_stage1_to_stage2(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            return
        table_path = self._find_first_file_by_suffix(project_path / "stage1", {".csv"})
        if table_path is None or not table_path.exists():
            messagebox.showwarning("Missing File", "Please save Stage 1 first.")
            return
        self.stage2_csv_path.delete(0, "end")
        self.stage2_csv_path.insert(0, str(table_path))
        self.load_stage2_table()
        self.tabview.set("Stage 2 - Analysis")

    def select_stage2_csv(self) -> None:
        selected = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not selected:
            return
        self.stage2_csv_path.delete(0, "end")
        self.stage2_csv_path.insert(0, selected)
        self.load_stage2_table()

    def load_stage2_table(self) -> None:
        df = self._load_stage2_data()
        if df is None:
            return
        self.stage2_loaded_df = df
        self._set_treeview_dataframe(self.stage2_table_tree, df)
        self._refresh_stage2_selection_menus()

    def _refresh_stage2_selection_menus(self) -> None:
        if self.stage2_loaded_df is None:
            return

        mode = self.stage2_extract_mode.get()
        if mode == "row":
            values = [str(i) for i in range(len(self.stage2_loaded_df.index))]
        else:
            values = [str(c) for c in self.stage2_loaded_df.columns]

        if not values:
            values = ["(none)"]

        yerr_values = ["(none)"] + values
        self.stage2_x_select.configure(values=values)
        self.stage2_y_select.configure(values=values)
        self.stage2_yerr_select.configure(values=yerr_values)

        self.stage2_x_select.set(values[0])
        self.stage2_y_select.set(values[min(1, len(values) - 1)])
        self.stage2_yerr_select.set("(none)")

    def _stage2_get_series(
        self,
        df: pd.DataFrame,
        mode: str,
        x_ref: str,
        y_ref: str,
        yerr_ref: Optional[str],
    ) -> Tuple[List[float], List[float], Optional[List[float]]]:
        if mode == "row":
            x_idx = int(x_ref)
            y_idx = int(y_ref)
            yerr_idx = int(yerr_ref) if yerr_ref else None

            numeric_df = df.apply(pd.to_numeric, errors="coerce")
            payload: Dict[str, Any] = {
                "x": numeric_df.iloc[x_idx, :],
                "y": numeric_df.iloc[y_idx, :],
            }
            if yerr_idx is not None:
                payload["yerr"] = numeric_df.iloc[yerr_idx, :]

            combo = pd.DataFrame(payload)
            subset = ["x", "y"] + (["yerr"] if "yerr" in combo.columns else [])
            combo = combo.dropna(subset=subset)
            if combo.empty:
                raise ValueError("No valid numeric pairs found for selected rows.")

            x_data = combo["x"].astype(float).tolist()
            y_data = combo["y"].astype(float).tolist()
            y_err = combo["yerr"].astype(float).tolist() if "yerr" in combo.columns else None
            return x_data, y_data, y_err

        payload_col: Dict[str, Any] = {
            "x": pd.to_numeric(df[x_ref], errors="coerce"),
            "y": pd.to_numeric(df[y_ref], errors="coerce"),
        }
        if yerr_ref:
            payload_col["yerr"] = pd.to_numeric(df[yerr_ref], errors="coerce")

        combo = pd.DataFrame(payload_col)
        subset = ["x", "y"] + (["yerr"] if "yerr" in combo.columns else [])
        combo = combo.dropna(subset=subset)
        if combo.empty:
            raise ValueError("No valid numeric pairs found for selected columns.")

        x_data = combo["x"].astype(float).tolist()
        y_data = combo["y"].astype(float).tolist()
        y_err = combo["yerr"].astype(float).tolist() if "yerr" in combo.columns else None
        return x_data, y_data, y_err

    def _stage2_apply_range(self, x_data: List[float], y_data: List[float], y_err: Optional[List[float]]) -> Tuple[List[float], List[float], Optional[List[float]]]:
        start_text = self.stage2_start_index.get().strip()
        end_text = self.stage2_end_index.get().strip()
        if not start_text and not end_text:
            return x_data, y_data, y_err

        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else len(x_data)
        if end <= start:
            raise ValueError("Data range end must be greater than start.")

        xs = x_data[start:end]
        ys = y_data[start:end]
        ye = y_err[start:end] if y_err is not None else None
        if len(xs) < 2 and self.stage2_method.get() != "fft":
            raise ValueError("Need at least 2 points after range slicing.")
        return xs, ys, ye

    def _stage2_plot_labels(self, x_ref: str, y_ref: str) -> Tuple[str, str, str]:
        mode = self.stage2_extract_mode.get()
        auto_x = self._auto_axis_label(x_ref, mode, "x")
        auto_y = self._auto_axis_label(y_ref, mode, "y")
        if auto_x == auto_y:
            auto_x = f"{auto_x}_x"
            auto_y = f"{auto_y}_y"

        default_title = f"{self.stage2_method.get().upper()} {auto_y} vs {auto_x}"
        default_x = auto_x
        default_y = auto_y

        title = self.stage2_plot_title.get().strip() or default_title
        xlabel = self.stage2_xlabel.get().strip() or default_x
        ylabel = self.stage2_ylabel.get().strip() or default_y
        return title, xlabel, ylabel

    def _load_stage2_data(self) -> Optional[pd.DataFrame]:
        csv_path = self.stage2_csv_path.get().strip()
        if not csv_path:
            messagebox.showwarning("Missing CSV", "Please select CSV input first.")
            return None
        if not Path(csv_path).exists():
            messagebox.showerror("CSV Not Found", csv_path)
            return None
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(how="all")
            return df
        except Exception as e:
            messagebox.showerror("Read CSV Error", str(e))
            return None

    def run_stage2_analysis(self) -> None:
        df = self.stage2_loaded_df if self.stage2_loaded_df is not None else self._load_stage2_data()
        if df is None:
            return

        method = self.stage2_method.get()
        mode = self.stage2_extract_mode.get()
        x_ref = self.stage2_x_select.get().strip()
        y_ref = self.stage2_y_select.get().strip()
        yerr_ref_raw = self.stage2_yerr_select.get().strip()
        yerr_ref = None if yerr_ref_raw in {"", "(none)"} else yerr_ref_raw

        if x_ref in {"", "(none)"} or y_ref in {"", "(none)"}:
            messagebox.showwarning("Missing Selection", "Please load table and choose X/Y from menu.")
            return

        title, xlabel, ylabel = self._stage2_plot_labels(x_ref, y_ref)

        try:
            self._set_task_status("Running Stage 2 analysis...")
            result: Dict[str, Any] = {"method": method}
            plot_b64 = ""

            if method == "fft":
                x_data, y_data, _ = self._stage2_get_series(df, mode, x_ref, y_ref, None)
                _, signal, _ = self._stage2_apply_range(x_data, y_data, None)
                sampling_rate = float(self.stage2_sampling_rate.get().strip() or "1.0")
                freq, magnitude = self.analyzer.fourier_transform(signal, sampling_rate=sampling_rate)
                result["frequency"] = [float(v) for v in freq.tolist()]
                result["magnitude"] = [float(v) for v in magnitude.tolist()]
                result["selection_mode"] = mode
                result["x_ref"] = x_ref
                result["y_ref"] = y_ref
                plot_b64 = self.analyzer.plot_fourier_transform(
                    signal,
                    sampling_rate=sampling_rate,
                    title=title,
                )
            else:
                x_data, y_data, y_err = self._stage2_get_series(df, mode, x_ref, y_ref, yerr_ref)
                x_data, y_data, y_err = self._stage2_apply_range(x_data, y_data, y_err)

                result["selection_mode"] = mode
                result["x_ref"] = x_ref
                result["y_ref"] = y_ref
                result["yerr_ref"] = yerr_ref
                result["used_points"] = len(x_data)
                result["plot_title"] = title
                result["xlabel"] = xlabel
                result["ylabel"] = ylabel

                if method == "linear":
                    slope, intercept, r2, slope_err, intercept_err, chi2_red = self.analyzer.linear_fit(x_data, y_data, y_err=y_err)
                    result.update(
                        {
                            "slope": slope,
                            "intercept": intercept,
                            "r_squared": r2,
                            "slope_err": slope_err,
                            "intercept_err": intercept_err,
                            "chi2_red": chi2_red,
                        }
                    )
                    plot_b64 = self.analyzer.plot_linear_fit(
                        x_data,
                        y_data,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        x_err=None,
                        y_err=y_err,
                        slope=slope,
                        intercept=intercept,
                        r_squared=r2,
                        slope_err=slope_err,
                        intercept_err=intercept_err,
                    )
                elif method == "log":
                    slope, intercept, r2, slope_err, intercept_err, chi2_red = self.analyzer.log_fit(x_data, y_data, y_err=y_err)
                    result.update(
                        {
                            "slope": slope,
                            "intercept": intercept,
                            "r_squared": r2,
                            "slope_err": slope_err,
                            "intercept_err": intercept_err,
                            "chi2_red": chi2_red,
                        }
                    )
                    plot_b64 = self.analyzer.plot_log_fit(
                        x_data,
                        y_data,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        x_err=None,
                        y_err=y_err,
                        slope=slope,
                        intercept=intercept,
                        r_squared=r2,
                        slope_err=slope_err,
                        intercept_err=intercept_err,
                    )
                elif method == "power":
                    k, c, r2, k_err, c_err, chi2_red = self.analyzer.power_fit(x_data, y_data, y_err=y_err)
                    result.update(
                        {
                            "k": k,
                            "c": c,
                            "r_squared": r2,
                            "k_err": k_err,
                            "c_err": c_err,
                            "chi2_red": chi2_red,
                        }
                    )
                    plot_b64 = self.analyzer.plot_power_fit(
                        x_data,
                        y_data,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        x_err=None,
                        y_err=y_err,
                        k=k,
                        c=c,
                        r_squared=r2,
                        k_err=k_err,
                        c_err=c_err,
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

            self.stage2_result_box.delete("1.0", "end")
            self.stage2_result_box.insert("1.0", self._render_stage2_result_summary(result))
            self._preview_base64_plot(plot_b64)

            self._save_stage2_outputs(result, plot_b64)
            self._refresh_project_status()
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
        finally:
            self._set_task_status("Idle")

    def _render_stage2_result_summary(self, result: Dict[str, Any]) -> str:
        method = str(result.get("method", "")).lower()
        lines: List[str] = ["[Analysis Key Output]", f"Method: {method or 'unknown'}"]

        if method in {"linear", "log"}:
            lines.extend(
                [
                    f"Slope: {float(result.get('slope', 0.0)):.6g}",
                    f"Slope uncertainty: {float(result.get('slope_err', 0.0)):.3g}",
                    f"Intercept: {float(result.get('intercept', 0.0)):.6g}",
                    f"Intercept uncertainty: {float(result.get('intercept_err', 0.0)):.3g}",
                    f"R^2: {float(result.get('r_squared', 0.0)):.6g}",
                    f"Reduced chi^2: {float(result.get('chi2_red', 0.0)):.6g}",
                    f"Used points: {int(result.get('used_points', 0) or 0)}",
                ]
            )
        elif method == "power":
            lines.extend(
                [
                    f"Exponent k: {float(result.get('k', 0.0)):.6g}",
                    f"k uncertainty: {float(result.get('k_err', 0.0)):.3g}",
                    f"Coefficient c: {float(result.get('c', 0.0)):.6g}",
                    f"c uncertainty: {float(result.get('c_err', 0.0)):.3g}",
                    f"R^2: {float(result.get('r_squared', 0.0)):.6g}",
                    f"Reduced chi^2: {float(result.get('chi2_red', 0.0)):.6g}",
                    f"Used points: {int(result.get('used_points', 0) or 0)}",
                ]
            )
        elif method == "fft":
            freq = result.get("frequency", [])
            mag = result.get("magnitude", [])
            peak_text = "N/A"
            try:
                if isinstance(freq, list) and isinstance(mag, list) and freq and mag and len(freq) == len(mag):
                    max_idx = max(range(len(mag)), key=lambda i: float(mag[i]))
                    peak_text = f"{float(freq[max_idx]):.6g} (magnitude={float(mag[max_idx]):.6g})"
            except Exception:
                peak_text = "N/A"

            lines.extend(
                [
                    f"Frequency bins: {len(freq) if isinstance(freq, list) else 0}",
                    f"Peak frequency: {peak_text}",
                ]
            )
        else:
            lines.append("No concise formatter for this method.")

        if bool(self.config_data.get("professional_mode", False)):
            lines.append("")
            lines.append("[Professional Mode: Full Data]")
            lines.append(json.dumps(result, ensure_ascii=False, indent=2))

        return "\n".join(lines)

    def _preview_base64_plot(self, plot_b64: str) -> None:
        if not plot_b64:
            self.stage2_plot_label.configure(text="(no plot)", image=None)
            return

        image_data = base64.b64decode(plot_b64)
        pil_image = Image.open(io.BytesIO(image_data))
        pil_image.thumbnail((840, 360))
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
        self.current_analysis_image_ctk = ctk_image
        self.stage2_plot_label.configure(image=ctk_image, text="")

    def _save_stage2_outputs(self, result: Dict[str, Any], plot_b64: str) -> None:
        project_path = self.get_project_path()
        if not project_path:
            return

        stage2_dir = project_path / "stage2"
        stage2_dir.mkdir(parents=True, exist_ok=True)

        with open(stage2_dir / "analysis_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        csv_input = self.stage2_csv_path.get().strip()
        if csv_input and Path(csv_input).exists():
            with open(stage2_dir / "analysis_input.csv", "w", encoding="utf-8") as out:
                out.write(Path(csv_input).read_text(encoding="utf-8", errors="ignore"))

        if plot_b64:
            default_stem = self._default_stage2_plot_stem(result)
            stem = self._sanitize_file_stem(self.stage2_plot_name_entry.get(), default_stem)
            plot_path = self._unique_path(stage2_dir / "plots" / f"{stem}.png")
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_bytes(base64.b64decode(plot_b64))

    def push_stage2_to_stage4(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            return
        result_path = self._find_first_file_by_suffix(project_path / "stage2", {".json"}, name_contains="analysis_result")
        if result_path is None or not result_path.exists():
            messagebox.showwarning("Missing Stage 2 Result", "Run Stage 2 analysis first.")
            return

        content = result_path.read_text(encoding="utf-8")
        self.stage4_context_box.insert("end", "\n\n[Stage2 Analysis]\n")
        self.stage4_context_box.insert("end", content)
        self.tabview.set("Stage 4 - AI Writing")

    def run_stage3_uncertainty(self) -> None:
        formula = self.stage3_formula.get().strip()
        if not formula:
            messagebox.showwarning("Missing Formula", "Please enter formula first.")
            return

        try:
            measurements, table_df = self._parse_stage3_measurements_table_from_tree()
        except Exception as e:
            messagebox.showerror("Table Error", f"Invalid measurements table: {e}")
            return

        ok, errs = validate_measurement_data(measurements)
        if not ok:
            messagebox.showwarning("Validation Warning", "\n".join(errs))

        try:
            self._set_task_status("Computing uncertainty...")
            calc = UncertaintyCalculator()
            expr = formula.split("=")[-1].strip() if "=" in formula else formula
            calc.parse_formula(expr)
            result = calc.compute_uncertainty_propagation_analytical(measurements)

            self.stage3_result_box.delete("1.0", "end")
            self.stage3_result_box.insert("1.0", self._render_stage3_result_summary(formula, result, measurements))

            project_path = self.get_project_path()
            if project_path:
                stage3_dir = project_path / "stage3"
                stage3_dir.mkdir(parents=True, exist_ok=True)
                with open(stage3_dir / "uncertainty_input.json", "w", encoding="utf-8") as f:
                    json.dump({"formula": formula, "measurements": measurements}, f, ensure_ascii=False, indent=2)
                table_df.to_csv(stage3_dir / "measurements_table.csv", index=False, encoding="utf-8")
                with open(stage3_dir / "uncertainty_result.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                self._refresh_project_status()
        except Exception as e:
            messagebox.showerror("Uncertainty Error", str(e))
        finally:
            self._set_task_status("Idle")

    def _render_stage3_result_summary(
        self,
        formula: str,
        result: Dict[str, Any],
        measurements: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        expr_name = formula.split("=")[0].strip() if "=" in formula else "Result"
        result_value = float(result.get("result", 0.0) or 0.0)
        u_total = float(result.get("uncertainty_total", 0.0) or 0.0)
        u_a = float(result.get("uncertainty_a", 0.0) or 0.0)
        u_b = float(result.get("uncertainty_b", 0.0) or 0.0)
        rel_u = float(result.get("relative_uncertainty", 0.0) or 0.0)

        value_fmt, unc_fmt = self.analyzer.format_with_uncertainty(result_value, u_total, sig=2)
        standard_form = f"{expr_name} = ({value_fmt} +- {unc_fmt})"

        lines = [
            "[Uncertainty Key Output]",
            f"Formula: {result.get('formula', '')}",
            f"Main result: {result_value:.6g}",
            f"Total uncertainty: {u_total:.4g}",
            f"A-type uncertainty: {u_a:.4g}",
            f"B-type uncertainty: {u_b:.4g}",
            f"Relative uncertainty: {rel_u:.2%}",
            "",
            "[Standard Form]",
            standard_form,
        ]

        if measurements:
            lines.append("")
            lines.append("[Measured Quantities]")
            for name, data in measurements.items():
                value = float(data.get("value", 0.0) or 0.0)
                unit = str(data.get("unit", "") or "")
                ua = float(data.get("a_uncertainty", 0.0) or 0.0)
                ub = float(data.get("b_uncertainty", 0.0) or 0.0)
                ut = (ua ** 2 + ub ** 2) ** 0.5
                lines.append(f"- {name} = {value:.6g} {unit} ; u_A={ua:.3g}, u_B={ub:.3g}, u={ut:.3g}")

        contributions = result.get("contributions", {})
        if isinstance(contributions, dict) and contributions:
            lines.append("")
            lines.append("[Top Contributions]")
            top_items = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:3]
            for var, pct in top_items:
                lines.append(f"- {var}: {float(pct):.1f}%")

        if bool(self.config_data.get("professional_mode", False)):
            lines.append("")
            lines.append("[Professional Mode: Full Data]")
            lines.append(json.dumps(result, ensure_ascii=False, indent=2))

        return "\n".join(lines)

    def _parse_stage3_measurements_table_from_tree(self) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
        df = self._treeview_to_dataframe(self.stage3_table_tree)
        if df.empty:
            raise ValueError("measurement table is empty")

        required_cols = {"quantity", "value", "unit", "a_uncertainty", "b_uncertainty"}
        miss = [c for c in required_cols if c not in df.columns]
        if miss:
            raise ValueError(f"missing required columns: {', '.join(miss)}")

        measurements: Dict[str, Dict[str, float]] = {}
        for _, row in df.iterrows():
            qty = str(row["quantity"]).strip()
            if not qty:
                continue
            measurements[qty] = {
                "value": float(row["value"]),
                "unit": str(row["unit"]).strip(),
                "a_uncertainty": float(row["a_uncertainty"]),
                "b_uncertainty": float(row["b_uncertainty"]),
            }

        if not measurements:
            raise ValueError("no valid quantity rows found")
        return measurements, df

    def _collect_project_context(self) -> str:
        project_path = self.get_project_path()
        folder_path: Optional[Path] = project_path if project_path and project_path.exists() else self.current_project_root
        if not folder_path:
            return self.stage4_context_box.get("1.0", "end").strip()

        if not self.stage4_file_items:
            self.stage4_file_items = self._scan_data_file_items(folder_path)
            self._populate_stage4_file_selectors(self.stage4_file_items)

        selected_items = self._get_selected_stage4_file_items()
        blocks: List[str] = ["[Selected Data Names By Type]\n" + self._format_stage4_selected_names(selected_items)]
        selected_data = self._build_stage4_selected_data_block(selected_items)
        if selected_data:
            blocks.append("[Selected Data Content]\n" + selected_data)

        manual = self.stage4_context_box.get("1.0", "end").strip()
        if manual:
            blocks.append("[Manual Context]\n" + manual)
        return "\n\n".join(blocks)

    def _scan_data_names_text(self, folder: Path) -> str:
        return self._format_stage4_selected_names(self._scan_data_file_items(folder))

    def _file_type_from_suffix(self, suffix: str) -> str:
        if suffix == ".csv":
            return "table"
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix == ".json":
            return "json"
        if suffix == ".tex":
            return "latex"
        if suffix == ".md":
            return "markdown"
        return "other"

    def _hash_file_for_dedup(self, path: Path) -> str:
        # Use a fast content fingerprint to avoid full-file hashing stalls on large files.
        h = hashlib.sha1()
        stat = path.stat()
        h.update(str(stat.st_size).encode("utf-8"))
        with open(path, "rb") as f:
            head = f.read(64 * 1024)
            h.update(head)
            if stat.st_size > 64 * 1024:
                try:
                    f.seek(max(0, stat.st_size - 64 * 1024))
                    tail = f.read(64 * 1024)
                    h.update(tail)
                except Exception:
                    pass
        return h.hexdigest()

    def _scan_data_file_items(self, folder: Path) -> List[Dict[str, Any]]:
        if not folder.exists() or not folder.is_dir():
            return []

        items: List[Dict[str, Any]] = []
        seen_keys: set[Tuple[str, str]] = set()

        for p in sorted(folder.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(folder))
            suffix = p.suffix.lower()
            ftype = self._file_type_from_suffix(suffix)

            dedup_id = rel.lower()
            if ftype == "table":
                try:
                    # Deduplicate copied stage2 input tables that are content-identical to stage1 tables.
                    dedup_id = self._hash_file_for_dedup(p)
                except Exception:
                    dedup_id = rel.lower()

            key = (ftype, dedup_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            items.append({"path": p, "rel": rel, "type": ftype})

        return items

    def _format_stage4_selected_names(self, items: List[Dict[str, Any]]) -> str:
        buckets: Dict[str, List[str]] = {
            "table": [],
            "image": [],
            "json": [],
            "latex": [],
            "markdown": [],
            "other": [],
        }
        for item in items:
            t = str(item.get("type", "other"))
            r = str(item.get("rel", ""))
            if t not in buckets:
                t = "other"
            buckets[t].append(r)

        lines: List[str] = []
        for key in ["table", "image", "json", "latex", "markdown", "other"]:
            values = buckets[key]
            lines.append(f"[{key}] ({len(values)})")
            if values:
                lines.extend([f"- {v}" for v in values])
        return "\n".join(lines)

    def _populate_stage4_file_selectors(self, items: List[Dict[str, Any]]) -> None:
        self.stage4_file_vars.clear()
        for child in self.stage4_file_select_frame.winfo_children():
            child.destroy()

        grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ["table", "image", "json", "latex", "markdown", "other"]}
        for item in items:
            t = str(item.get("type", "other"))
            if t not in grouped:
                t = "other"
            grouped[t].append(item)

        row = 0
        for key in ["table", "image", "json", "latex", "markdown", "other"]:
            ctk.CTkLabel(self.stage4_file_select_frame, text=f"[{key}]", anchor="w").grid(
                row=row, column=0, padx=4, pady=(4, 2), sticky="w"
            )
            row += 1
            for item in grouped[key]:
                rel = str(item.get("rel", ""))
                var = tk.BooleanVar(value=True)
                self.stage4_file_vars[rel] = var
                cb = ctk.CTkCheckBox(
                    self.stage4_file_select_frame,
                    text=rel,
                    variable=var,
                    onvalue=True,
                    offvalue=False,
                    command=self._sync_stage4_context_from_checks,
                )
                cb.grid(row=row, column=0, padx=18, pady=2, sticky="w")
                row += 1

    def _get_selected_stage4_file_items(self) -> List[Dict[str, Any]]:
        if not self.stage4_file_items:
            return []
        selected: List[Dict[str, Any]] = []
        for item in self.stage4_file_items:
            rel = str(item.get("rel", ""))
            var = self.stage4_file_vars.get(rel)
            if var is None or bool(var.get()):
                selected.append(item)
        return selected

    def _build_stage4_selected_data_block(self, items: List[Dict[str, Any]]) -> str:
        sections: List[str] = []
        total_chars = 0
        used_files = 0
        for item in items:
            if used_files >= MAX_STAGE4_CONTEXT_FILES or total_chars >= MAX_STAGE4_CONTEXT_CHARS:
                break

            p = item.get("path")
            rel = str(item.get("rel", ""))
            ftype = str(item.get("type", "other"))
            if not isinstance(p, Path) or not p.exists() or not p.is_file():
                continue

            if ftype == "image":
                sections.append(f"## {rel}\n(image file selected)")
                continue

            if p.suffix.lower() not in TEXT_FILE_EXTENSIONS and ftype != "table":
                sections.append(f"## {rel}\n(binary/non-text file selected)")
                continue

            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if len(raw) > MAX_STAGE4_FILE_CHARS:
                raw = raw[:MAX_STAGE4_FILE_CHARS] + "\n... (truncated)"

            block = f"## {rel}\n{raw}"
            sections.append(block)
            total_chars += len(block)
            used_files += 1

        if len(items) > used_files:
            sections.append(
                f"## context_notice\nOnly {used_files} files are included in context to keep UI responsive."
            )

        return "\n\n".join(sections)

    def _sync_stage4_context_from_checks(self) -> None:
        # Keep manual context untouched; checkbox states are consumed when collecting AI context.
        return

    def _stage4_select_all_files(self) -> None:
        for var in self.stage4_file_vars.values():
            var.set(True)
        self._sync_stage4_context_from_checks()

    def _stage4_clear_all_files(self) -> None:
        for var in self.stage4_file_vars.values():
            var.set(False)
        self._sync_stage4_context_from_checks()

    def _render_stage2_context_block(self, stage2_result_path: Path) -> str:
        text = stage2_result_path.read_text(encoding="utf-8", errors="ignore")
        try:
            result = json.loads(text)
        except Exception:
            return "[Stage2 Analysis]\n" + text

        if bool(self.config_data.get("professional_mode", False)):
            return "[Stage2 Analysis]\n" + json.dumps(result, ensure_ascii=False, indent=2)
        return "[Stage2 Analysis]\n" + self._render_stage2_result_summary(result)

    def _render_stage3_context_block(self, stage3_input_path: Path, stage3_result_path: Path) -> str:
        result_text = stage3_result_path.read_text(encoding="utf-8", errors="ignore")
        if bool(self.config_data.get("professional_mode", False)):
            return "[Stage3 Uncertainty Result]\n" + result_text

        formula = ""
        measurements: Optional[Dict[str, Dict[str, float]]] = None
        try:
            input_payload = json.loads(stage3_input_path.read_text(encoding="utf-8", errors="ignore"))
            formula = str(input_payload.get("formula", ""))
            m = input_payload.get("measurements")
            if isinstance(m, dict):
                measurements = m
        except Exception:
            pass

        try:
            result_payload = json.loads(result_text)
        except Exception:
            return "[Stage3 Uncertainty Result]\n" + result_text

        summary = self._render_stage3_result_summary(formula or "Result", result_payload, measurements)
        return "[Stage3 Uncertainty Result]\n" + summary

    def load_all_stage_data_to_context(self) -> None:
        self._set_task_status("Loading context data...")
        project_path = self.get_project_path()
        folder = project_path if project_path and project_path.exists() else self.current_project_root
        if folder:
            self.stage4_file_items = self._scan_data_file_items(folder)
            self._populate_stage4_file_selectors(self.stage4_file_items)

        self.stage4_has_asked_questions = False

        self._append_chat_bubble("System", "Data loaded. Please confirm selected files, then click AI Ask Questions.")
        messagebox.showinfo("Context Loaded", "Loaded deduplicated files by type. All are selected by default.")
        self._set_task_status("Idle")

    def _set_stage4_llm_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for name in ["stage4_btn_load", "stage4_btn_ask", "stage4_btn_send", "stage4_btn_compose", "stage4_prompt_entry"]:
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except Exception:
                    pass

    def _stream_llm_to_chat_async(
        self,
        messages: List[Dict[str, str]],
        prefix: str = "Assistant",
        code_block: bool = False,
        on_done: Optional[Any] = None,
    ) -> None:
        if self.stage4_llm_busy:
            messagebox.showinfo("Busy", "AI is still responding. Please wait for current response to finish.")
            return

        processor = self._new_llm_processor()
        if processor is None:
            return

        self.stage4_llm_busy = True
        self._set_stage4_llm_controls(False)
        if self._selected_enable_thinking():
            self._set_task_status("思考中，请稍后...")
        else:
            self._set_task_status(f"AI is generating ({self._selected_model()})...")
        bubble = self._append_chat_bubble(prefix, "")
        self.stage4_streaming_label = bubble

        def worker() -> None:
            full_text = ""
            err_text = ""
            try:
                for chunk in processor.generate_text_stream(
                    prompt="",
                    messages=messages,
                    enable_thinking=self._selected_enable_thinking(),
                ):
                    if chunk.get("type") != "content":
                        continue
                    piece = chunk.get("text", "")
                    if not piece:
                        continue
                    full_text += piece
                    snapshot = full_text

                    def update_ui(text_snapshot: str = snapshot) -> None:
                        if code_block:
                            bubble.configure(text=f"{prefix}:\n```latex\n{text_snapshot}\n```")
                        else:
                            display_text = self._sanitize_markdown_asterisk(text_snapshot)
                            bubble.configure(text=f"{prefix}:\n{display_text}")
                        try:
                            self.stage4_chat_frame._parent_canvas.yview_moveto(1.0)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    self.after(0, update_ui)
            except Exception as e:
                err_text = str(e)

            def finalize() -> None:
                self.stage4_streaming_label = None
                self.stage4_llm_busy = False
                self._set_stage4_llm_controls(True)
                self._set_task_status("Idle")
                if err_text:
                    messagebox.showerror("LLM Stream Error", err_text)
                    return
                if on_done:
                    result_text = full_text.strip()
                    if not code_block:
                        result_text = self._sanitize_markdown_asterisk(result_text)
                    on_done(result_text)

            self.after(0, finalize)

        threading.Thread(target=worker, daemon=True).start()

    def push_stage3_to_stage4(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            return
        result_path = self._find_first_file_by_suffix(project_path / "stage3", {".json"}, name_contains="uncertainty_result")
        if result_path is None or not result_path.exists():
            messagebox.showwarning("Missing Stage 3 Result", "Run Stage 3 first.")
            return

        self.stage4_context_box.insert("end", "\n\n[Stage3 Uncertainty]\n")
        self.stage4_context_box.insert("end", result_path.read_text(encoding="utf-8"))
        self.tabview.set("Stage 4 - AI Writing")

    def send_stage4_message(self) -> None:
        user_prompt = self.stage4_prompt_entry.get().strip()
        if not user_prompt:
            return
        if self.stage4_llm_busy:
            messagebox.showinfo("Busy", "AI is still responding. Please wait.")
            return

        project_path = self.get_project_path()
        folder = project_path if project_path and project_path.exists() else self.current_project_root
        if folder and not self.stage4_file_items:
            self.stage4_file_items = self._scan_data_file_items(folder)
            self._populate_stage4_file_selectors(self.stage4_file_items)

        context = self._collect_project_context()
        self.chat_history.append({"role": "user", "content": user_prompt})
        self._append_chat_bubble("User", user_prompt)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a physics lab report assistant. First communicate with the user to clarify missing details "
                    "(experiment name, objective, setup, key conclusion, desired format). Do not output full report yet. "
                    "Use concise Chinese and ask targeted questions when information is missing. "
                    "Do not use markdown asterisk formatting like *text* or list bullets '* '."
                ),
            },
            {"role": "system", "content": f"Project context:\n{context}"},
        ]
        if self.stage4_pending_report.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Current LaTeX draft exists below. If user asks to revise, return updated LaTeX snippet only for changed parts.\n"
                        + self.stage4_pending_report
                    ),
                }
            )
        messages.extend(self.chat_history)

        def on_done(reply: str) -> None:
            if not reply:
                return
            self.chat_history.append({"role": "assistant", "content": reply})
            self.stage4_prompt_entry.delete(0, "end")

            p = self.get_project_path()
            if p:
                self.save_stage4_chat(silent=True)
                self._refresh_project_status()

        self._stream_llm_to_chat_async(messages, prefix="Assistant", on_done=on_done)

    def generate_stage4_latex_report(self) -> None:
        if self.stage4_llm_busy:
            messagebox.showinfo("Busy", "AI is still responding. Please wait.")
            return
        context = self._collect_project_context()
        if not context.strip():
            messagebox.showwarning("Missing Context", "Please load or input context first.")
            return

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a physics lab report assistant. Generate only valid LaTeX content for a lab report. "
                    "Use structure: title, objective, theory, data/analysis, uncertainty, discussion, conclusion. "
                    "Do not wrap with markdown fences. Keep units and numbers consistent with provided data."
                ),
            },
            {"role": "system", "content": f"All available data:\n{context}"},
        ]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": "请基于以上沟通与数据，输出完整实验报告LaTeX源码。"})

        self._append_chat_bubble("System", "Composing LaTeX draft...")

        def on_done(latex_text: str) -> None:
            if not latex_text:
                return
            self.stage4_pending_report = latex_text
            self.chat_history.append({"role": "assistant", "content": latex_text})

            project_path = self.get_project_path()
            if project_path:
                stage4_dir = project_path / "stage4"
                stage4_dir.mkdir(parents=True, exist_ok=True)
                default_stem = self._sanitize_file_stem(self.current_project_name or "draft", "draft") + "_latex"
                stem = self._sanitize_file_stem(self.stage4_tex_name_entry.get(), default_stem)
                tex_path = self._unique_path(stage4_dir / f"{stem}.tex")
                tex_path.write_text(latex_text, encoding="utf-8")
                self.save_stage4_chat(silent=True)
                self._refresh_project_status()
                messagebox.showinfo("LaTeX Ready", f"LaTeX report saved to:\n{tex_path}")

        self._stream_llm_to_chat_async(messages, prefix="LaTeX", code_block=True, on_done=on_done)

    def stage4_ai_ask_questions(self) -> None:
        if self.stage4_llm_busy:
            messagebox.showinfo("Busy", "AI is still responding. Please wait.")
            return

        project_path = self.get_project_path()
        folder = project_path if project_path and project_path.exists() else self.current_project_root
        if folder and not self.stage4_file_items:
            self.stage4_file_items = self._scan_data_file_items(folder)
            self._populate_stage4_file_selectors(self.stage4_file_items)

        context = self._collect_project_context()
        if not context.strip():
            messagebox.showwarning("Missing Context", "Please load or input context first.")
            return

        self._append_chat_bubble("System", "AI is reading materials and preparing questions...")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a physics lab report assistant. Read project materials first, then ask concise targeted questions in Chinese "
                    "to collect missing experiment details. Do not generate report now. "
                    "Do not use markdown asterisk formatting like *text* or list bullets '* '."
                ),
            },
            {"role": "system", "content": f"Project context:\n{context}"},
            {"role": "user", "content": "请先阅读材料并向我提问缺失信息。"},
        ]

        def on_done(reply: str) -> None:
            if not reply:
                return
            self.chat_history.append({"role": "assistant", "content": reply})
            self.stage4_has_asked_questions = True

        self._stream_llm_to_chat_async(messages, prefix="Assistant", on_done=on_done)

    def save_stage4_chat(self, silent: bool = False) -> None:
        project_path = self.get_project_path()
        if not project_path:
            if not silent:
                messagebox.showwarning("No Project", "Please create/open project first.")
            return

        stage4_dir = project_path / "stage4"
        stage4_dir.mkdir(parents=True, exist_ok=True)

        chat_path = stage4_dir / "chat_history.json"
        with open(chat_path, "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=2)

        if not silent:
            messagebox.showinfo("Saved", f"Saved chat history to:\n{chat_path}")

    def _render_stage4_draft_text(self) -> str:
        context = self.stage4_context_box.get("1.0", "end").strip()
        chat_lines = [f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in self.chat_history]
        chat = "\n".join(chat_lines)
        return f"# Physics Lab Report Draft\n\n## Context\n{context}\n\n## AI Discussion\n{chat}\n"

    def save_stage4_draft_md(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            messagebox.showwarning("No Project", "Please create/open project first.")
            return

        draft_path = project_path / "stage4" / "draft.md"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(self._render_stage4_draft_text(), encoding="utf-8")
        self._refresh_project_status()
        messagebox.showinfo("Saved", f"Saved markdown draft to:\n{draft_path}")

    def save_stage4_draft_tex(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            messagebox.showwarning("No Project", "Please create/open project first.")
            return

        if self.stage4_pending_report.strip():
            default_stem = self._sanitize_file_stem(self.current_project_name or "draft", "draft") + "_latex"
            stem = self._sanitize_file_stem(self.stage4_tex_name_entry.get(), default_stem)
            draft_path = self._unique_path(project_path / "stage4" / f"{stem}.tex")
            draft_path.parent.mkdir(parents=True, exist_ok=True)
            draft_path.write_text(self.stage4_pending_report, encoding="utf-8")
            self._refresh_project_status()
            messagebox.showinfo("Saved", f"Saved LaTeX draft to:\n{draft_path}")
            return

        context = self.stage4_context_box.get("1.0", "end").strip().replace("_", "\\_")
        chat_lines = [f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in self.chat_history]
        chat = "\n".join(chat_lines).replace("_", "\\_")
        tex = (
            "\\documentclass{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\begin{document}\n"
            "\\section*{Physics Lab Report Draft}\n"
            "\\subsection*{Context}\n"
            f"{context}\n"
            "\\subsection*{AI Discussion}\n"
            f"{chat}\n"
            "\\end{document}\n"
        )

        default_stem = self._sanitize_file_stem(self.current_project_name or "draft", "draft") + "_latex"
        stem = self._sanitize_file_stem(self.stage4_tex_name_entry.get(), default_stem)
        draft_path = self._unique_path(project_path / "stage4" / f"{stem}.tex")
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(tex, encoding="utf-8")
        self._refresh_project_status()
        messagebox.showinfo("Saved", f"Saved LaTeX draft to:\n{draft_path}")

    def change_theme(self, theme: str) -> None:
        ctk.set_appearance_mode(theme)
        self.config_data["theme"] = theme
        self._save_config()

    def open_settings_window(self, force_api_key: bool = False) -> ctk.CTkToplevel:
        win = ctk.CTkToplevel(self)
        win.title("Settings")
        win.geometry("820x460")
        win.grab_set()
        win.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(win, text="DASHSCOPE API Key").grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")
        api_entry = ctk.CTkEntry(win, show="*")
        api_entry.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 6), sticky="ew")
        api_entry.insert(0, self.config_data.get("api_key", ""))
        ctk.CTkLabel(win, text=f"Config File: {CONFIG_PATH}", wraplength=560, justify="left").grid(
            row=2, column=0, columnspan=2, padx=12, pady=(0, 10), sticky="w"
        )

        button_row = ctk.CTkFrame(win, fg_color="transparent")
        button_row.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 10), sticky="w")

        ctk.CTkLabel(win, text="Default Project Root").grid(row=4, column=0, padx=12, pady=(6, 4), sticky="w")
        root_entry = ctk.CTkEntry(win)
        root_entry.grid(row=5, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="ew")
        root_entry.insert(0, self.config_data.get("default_project_root", str(Path.home())))

        professional_var = tk.BooleanVar(value=bool(self.config_data.get("professional_mode", False)))
        professional_switch = ctk.CTkSwitch(
            win,
            text="Professional Mode (show complete data)",
            variable=professional_var,
            onvalue=True,
            offvalue=False,
        )
        professional_switch.grid(row=6, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="w")

        ctk.CTkLabel(win, text="LLM Model").grid(row=7, column=0, padx=12, pady=(0, 12), sticky="w")
        model_option = ctk.CTkOptionMenu(win, values=MODEL_OPTIONS)
        model_option.set(self._selected_model())
        model_option.grid(row=7, column=1, padx=12, pady=(0, 12), sticky="w")

        ctk.CTkLabel(win, text="Font Size").grid(row=8, column=0, padx=12, pady=(0, 12), sticky="w")
        font_values = [str(v) for v in range(12, 25)]
        font_size_option = ctk.CTkOptionMenu(win, values=font_values)
        font_size_option.set(str(self._font_size()))
        font_size_option.grid(row=8, column=1, padx=12, pady=(0, 12), sticky="w")

        ctk.CTkLabel(win, text="Wheel Speed").grid(row=9, column=0, padx=12, pady=(0, 12), sticky="w")
        wheel_values = [str(v) for v in range(1, 21)]
        wheel_speed_option = ctk.CTkOptionMenu(win, values=wheel_values)
        wheel_speed_option.set(str(self._wheel_speed()))
        wheel_speed_option.grid(row=9, column=1, padx=12, pady=(0, 12), sticky="w")

        def browse_root() -> None:
            selected = filedialog.askdirectory(initialdir=root_entry.get().strip() or str(Path.home()))
            if selected:
                root_entry.delete(0, "end")
                root_entry.insert(0, selected)

        def open_config_dir() -> None:
            APP_DIR.mkdir(parents=True, exist_ok=True)
            try:
                os.startfile(str(APP_DIR))
            except Exception:
                messagebox.showinfo("Config Directory", str(APP_DIR))

        ctk.CTkButton(button_row, text="Browse", width=90, command=browse_root).grid(row=0, column=0, padx=(0, 8), pady=0, sticky="w")
        ctk.CTkButton(button_row, text="Open Config Folder", width=140, command=open_config_dir).grid(
            row=0, column=1, padx=(8, 0), pady=0, sticky="w"
        )

        def save_settings() -> None:
            api_key = api_entry.get().strip()
            if force_api_key and not api_key:
                messagebox.showwarning("Missing API Key", "API Key is required.")
                return

            self.config_data["api_key"] = api_key
            self.config_data["default_project_root"] = root_entry.get().strip() or str(Path.home())
            self.config_data["professional_mode"] = bool(professional_var.get())
            self.config_data["font_size"] = int(font_size_option.get())
            self.config_data["selected_model"] = model_option.get().strip()
            self.config_data["wheel_speed"] = int(wheel_speed_option.get())
            self._save_config()
            self._init_treeview_styles()
            self._apply_text_fonts()
            messagebox.showinfo("Saved", "Settings updated.")
            win.destroy()

        ctk.CTkButton(win, text="Save Settings", command=save_settings).grid(
            row=10, column=1, padx=12, pady=16, sticky="e"
        )
        return win

    def _refresh_project_status(self) -> None:
        project_path = self.get_project_path()
        if not project_path or not project_path.exists():
            self.stage_status_label.configure(text="Stage status\nS1: pending\nS2: pending\nS3: pending\nS4: pending")
            if hasattr(self, "workflow_progress"):
                self.workflow_progress.set(0)
            return

        s1 = "done" if self._find_first_file_by_suffix(project_path / "stage1", {".csv"}) else "pending"
        s2 = "done" if self._find_first_file_by_suffix(project_path / "stage2", {".json"}, name_contains="analysis_result") else "pending"
        s3 = "done" if self._find_first_file_by_suffix(project_path / "stage3", {".json"}, name_contains="uncertainty_result") else "pending"
        s4 = "done" if self._find_first_file_by_suffix(project_path / "stage4", {".tex", ".json"}) else "pending"
        self.stage_status_label.configure(text=f"Stage status\nS1: {s1}\nS2: {s2}\nS3: {s3}\nS4: {s4}")
        done_count = sum(1 for x in [s1, s2, s3, s4] if x == "done")
        if hasattr(self, "workflow_progress"):
            self.workflow_progress.set(done_count / 4.0)

    def save_all(self) -> None:
        try:
            self.save_stage1()
        except Exception:
            pass
        try:
            result_text = self.stage2_result_box.get("1.0", "end").strip()
            if result_text:
                self._save_stage2_outputs(json.loads(result_text), "")
        except Exception:
            pass
        try:
            result_text = self.stage3_result_box.get("1.0", "end").strip()
            if result_text and self.get_project_path():
                stage3_dir = self.get_project_path() / "stage3"
                stage3_dir.mkdir(parents=True, exist_ok=True)
                (stage3_dir / "uncertainty_result.json").write_text(result_text, encoding="utf-8")
        except Exception:
            pass
        try:
            self.save_stage4_chat(silent=True)
            self.save_stage4_draft_md()
        except Exception:
            pass


if __name__ == "__main__":
    app = PhysicsLabApp()
    app.mainloop()
