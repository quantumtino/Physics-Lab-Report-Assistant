import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk
import pandas as pd
from PIL import Image
from tkinter import filedialog, messagebox

from backend_core import DataAnalyzer, LLMProcessor, UncertaintyCalculator, validate_measurement_data


APP_DIR = Path.home() / ".physics_lab_report_assistant"
CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_CONFIG = {
    "api_key": "",
    "theme": "dark",
    "default_project_root": str(Path.home()),
}


class PhysicsLabApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Physics Lab Report Assistant")
        self.geometry("1320x860")
        self.minsize(1100, 760)

        self.config_data = self._load_config()
        ctk.set_appearance_mode(self.config_data.get("theme", "dark"))
        ctk.set_default_color_theme("blue")

        self.current_project_root: Optional[Path] = None
        self.current_project_name: Optional[str] = None
        self.current_image_path: Optional[Path] = None
        self.current_analysis_image_ctk: Optional[ctk.CTkImage] = None
        self.stage2_loaded_df: Optional[pd.DataFrame] = None
        self.chat_history: List[Dict[str, str]] = []
        self.stage4_pending_report: str = ""

        self.analyzer = DataAnalyzer()

        self._build_layout()
        self._refresh_project_status()

        if not self.config_data.get("api_key", "").strip():
            self.after(300, self._show_first_launch_hint)

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

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(12, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text="Physics Lab Assistant",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(20, 10), sticky="w")

        ctk.CTkButton(self.sidebar, text="Select Project Folder", command=self.select_project_root).grid(
            row=1, column=0, padx=16, pady=6, sticky="ew"
        )
        ctk.CTkLabel(self.sidebar, text="Project Name").grid(row=2, column=0, padx=16, pady=(10, 2), sticky="w")
        self.project_name_entry = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. optics_exp_01")
        self.project_name_entry.grid(row=3, column=0, padx=16, pady=4, sticky="ew")
        ctk.CTkButton(self.sidebar, text="Create / Open Project", command=self.create_or_open_project).grid(
            row=4, column=0, padx=16, pady=8, sticky="ew"
        )

        self.project_root_label = ctk.CTkLabel(self.sidebar, text="Root: (not selected)", wraplength=260, justify="left")
        self.project_root_label.grid(row=5, column=0, padx=16, pady=(6, 2), sticky="w")
        self.project_path_label = ctk.CTkLabel(self.sidebar, text="Project: (not open)", wraplength=260, justify="left")
        self.project_path_label.grid(row=6, column=0, padx=16, pady=(2, 10), sticky="w")

        self.stage_status_label = ctk.CTkLabel(self.sidebar, text="Stage status\nS1: pending\nS2: pending\nS3: pending\nS4: pending", justify="left")
        self.stage_status_label.grid(row=7, column=0, padx=16, pady=8, sticky="w")

        ctk.CTkLabel(self.sidebar, text="Theme").grid(row=8, column=0, padx=16, pady=(16, 4), sticky="w")
        self.theme_option = ctk.CTkOptionMenu(self.sidebar, values=["dark", "light"], command=self.change_theme)
        self.theme_option.set(self.config_data.get("theme", "dark"))
        self.theme_option.grid(row=9, column=0, padx=16, pady=4, sticky="ew")

        ctk.CTkButton(self.sidebar, text="Settings", command=self.open_settings_window).grid(
            row=10, column=0, padx=16, pady=(12, 6), sticky="ew"
        )
        ctk.CTkButton(self.sidebar, text="Save All", command=self.save_all).grid(
            row=11, column=0, padx=16, pady=6, sticky="ew"
        )

        self.main_panel = ctk.CTkFrame(self)
        self.main_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_panel.grid_columnconfigure(0, weight=1)
        self.main_panel.grid_rowconfigure(1, weight=1)

        self.header_label = ctk.CTkLabel(
            self.main_panel,
            text="Four-Stage Workflow",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.header_label.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        self.tabview = ctk.CTkTabview(self.main_panel)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))

        self.tab_stage1 = self.tabview.add("Stage 1 - OCR")
        self.tab_stage2 = self.tabview.add("Stage 2 - Analysis")
        self.tab_stage3 = self.tabview.add("Stage 3 - Uncertainty")
        self.tab_stage4 = self.tabview.add("Stage 4 - AI Writing")

        self._build_stage1()
        self._build_stage2()
        self._build_stage3()
        self._build_stage4()

    def _build_stage1(self) -> None:
        self.tab_stage1.grid_columnconfigure(0, weight=1)
        self.tab_stage1.grid_columnconfigure(1, weight=1)
        self.tab_stage1.grid_rowconfigure(3, weight=1)

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

        self.stage1_table_box = ctk.CTkTextbox(self.tab_stage1, height=340)
        self.stage1_table_box.grid(row=3, column=0, columnspan=2, padx=10, pady=8, sticky="nsew")

        btn_row = ctk.CTkFrame(self.tab_stage1, fg_color="transparent")
        btn_row.grid(row=4, column=0, columnspan=2, padx=10, pady=8, sticky="ew")
        btn_row.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(btn_row, text="Save Stage 1", command=self.save_stage1).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(btn_row, text="Push to Stage 2", command=self.push_stage1_to_stage2).grid(
            row=0, column=1, padx=4, pady=4, sticky="ew"
        )

    def _build_stage2(self) -> None:
        self.tab_stage2.grid_columnconfigure(0, weight=1)
        self.tab_stage2.grid_columnconfigure(1, weight=1)
        self.tab_stage2.grid_rowconfigure(6, weight=1)

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

        ctk.CTkButton(self.tab_stage2, text="Run Analysis", command=self.run_stage2_analysis).grid(
            row=1, column=0, padx=10, pady=6, sticky="w"
        )
        ctk.CTkButton(self.tab_stage2, text="Push to Stage 4", command=self.push_stage2_to_stage4).grid(
            row=1, column=1, padx=10, pady=6, sticky="e"
        )

        self.stage2_result_box = ctk.CTkTextbox(self.tab_stage2, height=230)
        self.stage2_result_box.grid(row=2, column=0, columnspan=2, padx=10, pady=8, sticky="ew")

        self.stage2_plot_label = ctk.CTkLabel(self.tab_stage2, text="(plot preview)")
        self.stage2_plot_label.grid(row=6, column=0, columnspan=2, padx=10, pady=8, sticky="nsew")

    def _build_stage3(self) -> None:
        self.tab_stage3.grid_columnconfigure(0, weight=1)
        self.tab_stage3.grid_columnconfigure(1, weight=1)
        self.tab_stage3.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(self.tab_stage3, text="Formula").grid(row=0, column=0, padx=10, pady=(12, 4), sticky="w")
        self.stage3_formula = ctk.CTkEntry(self.tab_stage3, placeholder_text="e.g. F = m*a")
        self.stage3_formula.grid(row=0, column=1, padx=10, pady=(12, 4), sticky="ew")

        ctk.CTkLabel(self.tab_stage3, text="Measurements Table (CSV format)").grid(row=1, column=0, padx=10, pady=4, sticky="w")
        self.stage3_measure_box = ctk.CTkTextbox(self.tab_stage3, height=180)
        self.stage3_measure_box.grid(row=1, column=1, padx=10, pady=4, sticky="ew")
        self.stage3_measure_box.insert(
            "1.0",
            "quantity,value,unit,a_uncertainty,b_uncertainty\n"
            "m,0.5,kg,0.001,0.0005\n"
            "a,2.0,m/s^2,0.02,0.01\n",
        )

        ctk.CTkLabel(
            self.tab_stage3,
            text="Edit rows directly. Required columns: quantity,value,unit,a_uncertainty,b_uncertainty",
            anchor="w",
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 4), sticky="ew")

        ctk.CTkButton(self.tab_stage3, text="Compute Uncertainty", command=self.run_stage3_uncertainty).grid(
            row=3, column=0, padx=10, pady=6, sticky="w"
        )
        ctk.CTkButton(self.tab_stage3, text="Push to Stage 4", command=self.push_stage3_to_stage4).grid(
            row=3, column=1, padx=10, pady=6, sticky="e"
        )

        self.stage3_result_box = ctk.CTkTextbox(self.tab_stage3)
        self.stage3_result_box.grid(row=4, column=0, columnspan=2, padx=10, pady=8, sticky="nsew")

    def _build_stage4(self) -> None:
        self.tab_stage4.grid_columnconfigure(0, weight=1)
        self.tab_stage4.grid_columnconfigure(1, weight=1)
        self.tab_stage4.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(self.tab_stage4, text="Context (from Stage 1/2/3 + manual notes)").grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        self.stage4_context_box = ctk.CTkTextbox(self.tab_stage4, height=120)
        self.stage4_context_box.grid(row=1, column=0, columnspan=2, padx=10, pady=6, sticky="ew")

        top_actions = ctk.CTkFrame(self.tab_stage4, fg_color="transparent")
        top_actions.grid(row=0, column=1, padx=10, pady=(8, 4), sticky="e")
        ctk.CTkButton(top_actions, text="Load All Stage Data", command=self.load_all_stage_data_to_context).grid(
            row=0, column=0, padx=4, pady=4
        )
        ctk.CTkButton(top_actions, text="Generate LaTeX Report", command=self.generate_stage4_latex_report).grid(
            row=0, column=1, padx=4, pady=4
        )

        self.stage4_chat_box = ctk.CTkTextbox(self.tab_stage4)
        self.stage4_chat_box.grid(row=2, column=0, columnspan=2, padx=10, pady=6, sticky="nsew")

        self.stage4_prompt_entry = ctk.CTkEntry(self.tab_stage4, placeholder_text="Ask AI to draft/report/interpret result...")
        self.stage4_prompt_entry.grid(row=3, column=0, padx=10, pady=8, sticky="ew")
        ctk.CTkButton(self.tab_stage4, text="Send", command=self.send_stage4_message).grid(
            row=3, column=1, padx=10, pady=8, sticky="e"
        )

        action_bar = ctk.CTkFrame(self.tab_stage4, fg_color="transparent")
        action_bar.grid(row=4, column=0, columnspan=2, padx=10, pady=8, sticky="ew")
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

    def get_project_path(self) -> Optional[Path]:
        if not self.current_project_root or not self.current_project_name:
            return None
        return self.current_project_root / self.current_project_name

    def select_project_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.config_data.get("default_project_root", str(Path.home())))
        if not selected:
            return
        self.current_project_root = Path(selected)
        self.config_data["default_project_root"] = selected
        self._save_config()
        self.project_root_label.configure(text=f"Root: {selected}")

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
            return LLMProcessor()
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
            with open(self.current_image_path, "rb") as f:
                image_bytes = f.read()
            df = processor.extract_table_from_image_bytes(image_bytes, mime_type=mime_type)
            csv_text = df.to_csv(index=False)
            self.stage1_table_box.delete("1.0", "end")
            self.stage1_table_box.insert("1.0", csv_text)
            self.tabview.set("Stage 1 - OCR")
        except Exception as e:
            messagebox.showerror("OCR Error", str(e))

    def save_stage1(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            messagebox.showwarning("No Project", "Please create/open project first.")
            return

        csv_text = self.stage1_table_box.get("1.0", "end").strip()
        if not csv_text:
            messagebox.showwarning("No Table", "No OCR table content to save.")
            return

        stage1_dir = project_path / "stage1"
        table_path = stage1_dir / "table.csv"
        ocr_json_path = stage1_dir / "ocr_result.json"

        with open(table_path, "w", encoding="utf-8") as f:
            f.write(csv_text + "\n")
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump({"table_csv_path": str(table_path)}, f, ensure_ascii=False, indent=2)

        if self.current_image_path and self.current_image_path.exists():
            copied_path = stage1_dir / "ocr_input" / self.current_image_path.name
            if copied_path != self.current_image_path:
                copied_path.write_bytes(self.current_image_path.read_bytes())

        self._refresh_project_status()
        messagebox.showinfo("Saved", f"Stage 1 saved to:\n{table_path}")

    def push_stage1_to_stage2(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            return
        table_path = project_path / "stage1" / "table.csv"
        if not table_path.exists():
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
        default_title = f"{self.stage2_method.get().upper()} Fit ({mode})"
        default_x = f"{mode}:{x_ref}"
        default_y = f"{mode}:{y_ref}"

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
            self.stage2_result_box.insert("1.0", json.dumps(result, ensure_ascii=False, indent=2))
            self._preview_base64_plot(plot_b64)

            self._save_stage2_outputs(result, plot_b64)
            self._refresh_project_status()
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))

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
            plot_path = stage2_dir / "plots" / "latest_plot.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_bytes(base64.b64decode(plot_b64))

    def push_stage2_to_stage4(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            return
        result_path = project_path / "stage2" / "analysis_result.json"
        if not result_path.exists():
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
            measurements = self._parse_stage3_measurements_table(self.stage3_measure_box.get("1.0", "end"))
        except Exception as e:
            messagebox.showerror("Table Error", f"Invalid measurements table: {e}")
            return

        ok, errs = validate_measurement_data(measurements)
        if not ok:
            messagebox.showwarning("Validation Warning", "\n".join(errs))

        try:
            calc = UncertaintyCalculator()
            expr = formula.split("=")[-1].strip() if "=" in formula else formula
            calc.parse_formula(expr)
            result = calc.compute_uncertainty_propagation_analytical(measurements)

            self.stage3_result_box.delete("1.0", "end")
            self.stage3_result_box.insert("1.0", json.dumps(result, ensure_ascii=False, indent=2))

            project_path = self.get_project_path()
            if project_path:
                stage3_dir = project_path / "stage3"
                stage3_dir.mkdir(parents=True, exist_ok=True)
                with open(stage3_dir / "uncertainty_input.json", "w", encoding="utf-8") as f:
                    json.dump({"formula": formula, "measurements": measurements}, f, ensure_ascii=False, indent=2)
                with open(stage3_dir / "measurements_table.csv", "w", encoding="utf-8") as f:
                    f.write(self.stage3_measure_box.get("1.0", "end").strip() + "\n")
                with open(stage3_dir / "uncertainty_result.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                self._refresh_project_status()
        except Exception as e:
            messagebox.showerror("Uncertainty Error", str(e))

    def _parse_stage3_measurements_table(self, table_text: str) -> Dict[str, Dict[str, float]]:
        raw = table_text.strip()
        if not raw:
            raise ValueError("measurement table is empty")

        df = pd.read_csv(io.StringIO(raw))
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
        return measurements

    def _collect_project_context(self) -> str:
        project_path = self.get_project_path()
        if not project_path:
            return self.stage4_context_box.get("1.0", "end").strip()

        blocks: List[str] = []
        stage1_table = project_path / "stage1" / "table.csv"
        stage2_result = project_path / "stage2" / "analysis_result.json"
        stage3_input = project_path / "stage3" / "uncertainty_input.json"
        stage3_result = project_path / "stage3" / "uncertainty_result.json"

        if stage1_table.exists():
            blocks.append("[Stage1 Table CSV]\n" + stage1_table.read_text(encoding="utf-8", errors="ignore"))
        if stage2_result.exists():
            blocks.append("[Stage2 Analysis JSON]\n" + stage2_result.read_text(encoding="utf-8", errors="ignore"))
        if stage3_input.exists():
            blocks.append("[Stage3 Uncertainty Input]\n" + stage3_input.read_text(encoding="utf-8", errors="ignore"))
        if stage3_result.exists():
            blocks.append("[Stage3 Uncertainty Result]\n" + stage3_result.read_text(encoding="utf-8", errors="ignore"))

        manual = self.stage4_context_box.get("1.0", "end").strip()
        if manual:
            blocks.append("[Manual Context]\n" + manual)
        return "\n\n".join(blocks)

    def load_all_stage_data_to_context(self) -> None:
        context = self._collect_project_context()
        self.stage4_context_box.delete("1.0", "end")
        self.stage4_context_box.insert("1.0", context)
        messagebox.showinfo("Context Loaded", "Loaded available Stage 1/2/3 data into context.")

    def _stream_llm_to_chat(self, messages: List[Dict[str, str]], prefix: str = "Assistant") -> str:
        processor = self._new_llm_processor()
        if processor is None:
            return ""

        self.stage4_chat_box.insert("end", f"\n{prefix}: ")
        self.stage4_chat_box.see("end")
        self.update_idletasks()

        full_text = ""
        try:
            for chunk in processor.generate_text_stream(prompt="", messages=messages):
                if chunk.get("type") != "content":
                    continue
                piece = chunk.get("text", "")
                if not piece:
                    continue
                full_text += piece
                self.stage4_chat_box.insert("end", piece)
                self.stage4_chat_box.see("end")
                self.update_idletasks()
        except Exception as e:
            messagebox.showerror("LLM Stream Error", str(e))
            return ""

        self.stage4_chat_box.insert("end", "\n")
        self.stage4_chat_box.see("end")
        return full_text.strip()

    def push_stage3_to_stage4(self) -> None:
        project_path = self.get_project_path()
        if not project_path:
            return
        result_path = project_path / "stage3" / "uncertainty_result.json"
        if not result_path.exists():
            messagebox.showwarning("Missing Stage 3 Result", "Run Stage 3 first.")
            return

        self.stage4_context_box.insert("end", "\n\n[Stage3 Uncertainty]\n")
        self.stage4_context_box.insert("end", result_path.read_text(encoding="utf-8"))
        self.tabview.set("Stage 4 - AI Writing")

    def send_stage4_message(self) -> None:
        user_prompt = self.stage4_prompt_entry.get().strip()
        if not user_prompt:
            return

        context = self._collect_project_context()
        self.chat_history.append({"role": "user", "content": user_prompt})
        self.stage4_chat_box.insert("end", f"\nUser: {user_prompt}\n")
        self.stage4_chat_box.see("end")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a physics lab report assistant. First communicate with the user to clarify missing details "
                    "(experiment name, objective, setup, key conclusion, desired format). Do not output full report yet. "
                    "Use concise Chinese and ask targeted questions when information is missing."
                ),
            },
            {"role": "system", "content": f"Project context:\n{context}"},
        ]
        messages.extend(self.chat_history)

        reply = self._stream_llm_to_chat(messages)
        if reply:
            self.chat_history.append({"role": "assistant", "content": reply})
            self.stage4_prompt_entry.delete(0, "end")

            project_path = self.get_project_path()
            if project_path:
                self.save_stage4_chat(silent=True)
                self._refresh_project_status()

    def generate_stage4_latex_report(self) -> None:
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

        self.stage4_chat_box.insert("end", "\n[Generating LaTeX report...]\n")
        self.stage4_chat_box.see("end")

        latex_text = self._stream_llm_to_chat(messages, prefix="LaTeX")
        if not latex_text:
            return

        self.stage4_pending_report = latex_text
        self.chat_history.append({"role": "assistant", "content": latex_text})

        project_path = self.get_project_path()
        if project_path:
            stage4_dir = project_path / "stage4"
            stage4_dir.mkdir(parents=True, exist_ok=True)
            (stage4_dir / "draft.tex").write_text(latex_text, encoding="utf-8")
            self.save_stage4_chat(silent=True)
            self._refresh_project_status()
            messagebox.showinfo("LaTeX Ready", f"LaTeX report saved to:\n{stage4_dir / 'draft.tex'}")

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
        chat = self.stage4_chat_box.get("1.0", "end").strip()
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
            draft_path = project_path / "stage4" / "draft.tex"
            draft_path.parent.mkdir(parents=True, exist_ok=True)
            draft_path.write_text(self.stage4_pending_report, encoding="utf-8")
            self._refresh_project_status()
            messagebox.showinfo("Saved", f"Saved LaTeX draft to:\n{draft_path}")
            return

        context = self.stage4_context_box.get("1.0", "end").strip().replace("_", "\\_")
        chat = self.stage4_chat_box.get("1.0", "end").strip().replace("_", "\\_")
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

        draft_path = project_path / "stage4" / "draft.tex"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(tex, encoding="utf-8")
        self._refresh_project_status()
        messagebox.showinfo("Saved", f"Saved LaTeX draft to:\n{draft_path}")

    def change_theme(self, theme: str) -> None:
        ctk.set_appearance_mode(theme)
        self.config_data["theme"] = theme
        self._save_config()

    def open_settings_window(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Settings")
        win.geometry("700x320")
        win.grab_set()
        win.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(win, text="DASHSCOPE API Key").grid(row=0, column=0, padx=12, pady=12, sticky="w")
        api_entry = ctk.CTkEntry(win)
        api_entry.grid(row=0, column=1, padx=12, pady=12, sticky="ew")
        api_entry.insert(0, self.config_data.get("api_key", ""))

        ctk.CTkLabel(win, text="Default Project Root").grid(row=1, column=0, padx=12, pady=12, sticky="w")
        root_entry = ctk.CTkEntry(win)
        root_entry.grid(row=1, column=1, padx=12, pady=12, sticky="ew")
        root_entry.insert(0, self.config_data.get("default_project_root", str(Path.home())))

        def browse_root() -> None:
            selected = filedialog.askdirectory(initialdir=root_entry.get().strip() or str(Path.home()))
            if selected:
                root_entry.delete(0, "end")
                root_entry.insert(0, selected)

        ctk.CTkButton(win, text="Browse", width=90, command=browse_root).grid(row=1, column=2, padx=8, pady=12)

        def save_settings() -> None:
            self.config_data["api_key"] = api_entry.get().strip()
            self.config_data["default_project_root"] = root_entry.get().strip() or str(Path.home())
            self._save_config()
            messagebox.showinfo("Saved", "Settings updated.")
            win.destroy()

        ctk.CTkButton(win, text="Save Settings", command=save_settings).grid(
            row=2, column=1, padx=12, pady=16, sticky="e"
        )

    def _refresh_project_status(self) -> None:
        project_path = self.get_project_path()
        if not project_path or not project_path.exists():
            self.stage_status_label.configure(text="Stage status\nS1: pending\nS2: pending\nS3: pending\nS4: pending")
            return

        s1 = "done" if (project_path / "stage1" / "table.csv").exists() else "pending"
        s2 = "done" if (project_path / "stage2" / "analysis_result.json").exists() else "pending"
        s3 = "done" if (project_path / "stage3" / "uncertainty_result.json").exists() else "pending"
        s4 = "done" if (project_path / "stage4" / "chat_history.json").exists() else "pending"
        self.stage_status_label.configure(text=f"Stage status\nS1: {s1}\nS2: {s2}\nS3: {s3}\nS4: {s4}")

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
