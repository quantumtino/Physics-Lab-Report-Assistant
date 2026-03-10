import base64
import io
import json
import math
import os
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class SymbolicMathTools:
    @staticmethod
    def compute_partial_derivative(expression: str, variable: str) -> Dict[str, Any]:
        try:
            expression = expression.replace("^", "**").replace("ln", "log")
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            derivative = sp.diff(expr, var)
            simplified = sp.simplify(derivative)
            return {
                "success": True,
                "derivative": str(derivative),
                "derivative_latex": sp.latex(derivative),
                "simplified": str(simplified),
                "simplified_latex": sp.latex(simplified),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def evaluate_expression(expression: str, substitutions: Dict[str, float]) -> Dict[str, Any]:
        try:
            expression = expression.replace("^", "**").replace("ln", "log")
            expr = sp.sympify(expression)
            subs_dict = {sp.Symbol(k): v for k, v in substitutions.items()}
            result = float(expr.subs(subs_dict))
            return {"success": True, "value": result, "expression": str(expr)}
        except Exception as e:
            return {"success": False, "error": str(e)}


def call_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    tools = SymbolicMathTools()
    if tool_name == "compute_partial_derivative":
        return tools.compute_partial_derivative(kwargs["expression"], kwargs["variable"])
    if tool_name == "evaluate_expression":
        return tools.evaluate_expression(kwargs["expression"], kwargs["substitutions"])
    return {"success": False, "error": f"Unknown tool: {tool_name}"}


class UncertaintyCalculator:
    def __init__(self):
        self.formula = None
        self.variables = {}

    def parse_formula(self, formula_str: str) -> sp.Expr:
        formula_str = formula_str.replace("^", "**").replace("ln", "log")
        expr = sp.sympify(formula_str)
        self.formula = expr
        self.variables = {str(sym): None for sym in expr.free_symbols}
        return expr

    def compute_uncertainty_propagation_analytical(self, measurements: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        if self.formula is None:
            raise ValueError("Formula is not parsed")

        partial_derivs_sym = {}
        for var_str in measurements.keys():
            var = sp.Symbol(var_str)
            partial_derivs_sym[var] = sp.diff(self.formula, var)

        subs_dict = {sp.Symbol(k): v["value"] for k, v in measurements.items()}

        partial_values = {}
        for var, partial in partial_derivs_sym.items():
            try:
                partial_values[str(var)] = float(partial.subs(subs_dict))
            except Exception:
                partial_values[str(var)] = 0.0

        uncertainty_sq_a = 0.0
        uncertainty_sq_b = 0.0
        contributions = {}
        contributions_detailed = {}

        for var_str, data in measurements.items():
            partial_val = partial_values[var_str]
            u_a = float(data.get("a_uncertainty", 0) or 0)
            u_b = float(data.get("b_uncertainty", 0) or 0)

            contrib_a = (partial_val * u_a) ** 2
            contrib_b = (partial_val * u_b) ** 2
            contrib_total = contrib_a + contrib_b

            uncertainty_sq_a += contrib_a
            uncertainty_sq_b += contrib_b

            contributions_detailed[var_str] = {
                "a_contribution": float(contrib_a),
                "b_contribution": float(contrib_b),
                "total_contribution": float(contrib_total),
                "partial_derivative": float(partial_val),
                "a_uncertainty": u_a,
                "b_uncertainty": u_b,
            }

        uncertainty_a = float(np.sqrt(uncertainty_sq_a))
        uncertainty_b = float(np.sqrt(uncertainty_sq_b))
        uncertainty_total = float(np.sqrt(uncertainty_sq_a + uncertainty_sq_b))

        total_sq = uncertainty_sq_a + uncertainty_sq_b
        for var_str in measurements.keys():
            if total_sq > 0:
                contributions[var_str] = (
                    contributions_detailed[var_str]["total_contribution"] / total_sq * 100
                )
            else:
                contributions[var_str] = 0.0

        result = float(self.formula.subs(subs_dict))

        return {
            "method": "analytical",
            "result": result,
            "uncertainty_a": uncertainty_a,
            "uncertainty_b": uncertainty_b,
            "uncertainty_total": uncertainty_total,
            "relative_uncertainty": uncertainty_total / result if result != 0 else float("inf"),
            "partial_derivatives": {
                str(var): {
                    "expression": str(partial),
                    "latex": sp.latex(partial),
                    "value": partial_values[str(var)],
                }
                for var, partial in partial_derivs_sym.items()
            },
            "contributions": contributions,
            "contributions_detailed": contributions_detailed,
            "formula": str(self.formula),
            "formula_latex": sp.latex(self.formula),
        }


def validate_measurement_data(measurements: Dict[str, Dict[str, float]]) -> Tuple[bool, List[str]]:
    errors = []
    for var_str, data in measurements.items():
        value = data.get("value", 0)
        a_unc = data.get("a_uncertainty", 0)
        b_unc = data.get("b_uncertainty", 0)

        total_unc = np.sqrt(a_unc ** 2 + b_unc ** 2)
        if value != 0:
            rel_unc = total_unc / abs(value)
            if rel_unc > 0.5:
                errors.append(f"{var_str}: relative uncertainty is too large ({rel_unc:.1%})")
            elif rel_unc < 0.001:
                errors.append(f"{var_str}: relative uncertainty is too small ({rel_unc:.2%})")

        if a_unc < 0 or b_unc < 0:
            errors.append(f"{var_str}: uncertainty cannot be negative")

        if "unit" not in data or not data["unit"]:
            errors.append(f"{var_str}: unit is missing")

    return len(errors) == 0, errors


class DataAnalyzer:
    def format_with_uncertainty(self, value: float, uncertainty: float, sig: int = 2) -> Tuple[str, str]:
        if uncertainty is None or uncertainty <= 0 or math.isnan(uncertainty):
            return f"{value:.4g}", "0"

        exp = math.floor(math.log10(abs(uncertainty)))
        dec = max(0, sig - 1 - exp)
        unc_rounded = round(uncertainty, dec)
        val_rounded = round(value, dec)
        fmt = f"{{:.{dec}f}}"
        return fmt.format(val_rounded), fmt.format(unc_rounded)

    def linear_fit(
        self,
        x_data: List[float],
        y_data: List[float],
        y_err: Optional[List[float]] = None,
        x_err: Optional[List[float]] = None,
    ) -> Tuple[float, float, float, float, float, float]:
        x = np.array(x_data)
        y = np.array(y_data)

        use_weights = y_err is not None and len(y_err) == len(y) and np.all(np.array(y_err) > 0)

        if use_weights:
            w = 1.0 / (np.array(y_err) ** 2)
            s = w.sum()
            sx = (w * x).sum()
            sy = (w * y).sum()
            sxx = (w * x * x).sum()
            sxy = (w * x * y).sum()
            delta = s * sxx - sx * sx

            slope = (s * sxy - sx * sy) / delta
            intercept = (sy - slope * sx) / s

            y_hat = slope * x + intercept
            y_bar_w = sy / s
            sse = (w * (y - y_hat) ** 2).sum()
            sst = (w * (y - y_bar_w) ** 2).sum()
            r_squared_w = 1.0 - (sse / sst if sst > 0 else 0.0)

            dof = max(1, len(x) - 2)
            chi2_red = sse / dof
            var_slope = s / delta
            var_intercept = sxx / delta
            slope_err = float(np.sqrt(var_slope * chi2_red))
            intercept_err = float(np.sqrt(var_intercept * chi2_red))
            return float(slope), float(intercept), float(r_squared_w), slope_err, intercept_err, float(chi2_red)

        (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
        slope_err = float(np.sqrt(cov[0, 0])) if cov.size else 0.0
        intercept_err = float(np.sqrt(cov[1, 1])) if cov.size else 0.0
        correlation_matrix = np.corrcoef(x, y)
        r_squared = float(correlation_matrix[0, 1] ** 2)

        y_hat = slope * x + intercept
        resid = y - y_hat
        dof = max(1, len(x) - 2)
        chi2_red = float((resid ** 2).sum() / dof)
        return float(slope), float(intercept), float(r_squared), slope_err, intercept_err, chi2_red

    def log_fit(
        self, x_data: List[float], y_data: List[float], y_err: Optional[List[float]] = None
    ) -> Tuple[float, float, float, float, float, float]:
        x = np.array(x_data)
        y = np.array(y_data)
        if np.any(x <= 0):
            raise ValueError("Log fit requires all x values to be > 0")

        x_log = np.log(x)
        use_weights = y_err is not None and len(y_err) == len(y) and np.all(np.array(y_err) > 0)

        if use_weights:
            w = 1.0 / (np.array(y_err) ** 2)
            s = w.sum()
            sx = (w * x_log).sum()
            sy = (w * y).sum()
            sxx = (w * x_log * x_log).sum()
            sxy = (w * x_log * y).sum()
            delta = s * sxx - sx * sx

            slope = (s * sxy - sx * sy) / delta
            intercept = (sy - slope * sx) / s

            y_hat = slope * x_log + intercept
            y_bar_w = sy / s
            sse = (w * (y - y_hat) ** 2).sum()
            sst = (w * (y - y_bar_w) ** 2).sum()
            r_squared_w = 1.0 - (sse / sst if sst > 0 else 0.0)

            dof = max(1, len(x_log) - 2)
            chi2_red = sse / dof
            var_slope = s / delta
            var_intercept = sxx / delta
            slope_err = float(np.sqrt(var_slope * chi2_red))
            intercept_err = float(np.sqrt(var_intercept * chi2_red))
            return float(slope), float(intercept), float(r_squared_w), slope_err, intercept_err, float(chi2_red)

        (slope, intercept), cov = np.polyfit(x_log, y, 1, cov=True)
        slope_err = float(np.sqrt(cov[0, 0])) if cov.size else 0.0
        intercept_err = float(np.sqrt(cov[1, 1])) if cov.size else 0.0
        correlation_matrix = np.corrcoef(x_log, y)
        r_squared = float(correlation_matrix[0, 1] ** 2)

        y_hat = slope * x_log + intercept
        resid = y - y_hat
        dof = max(1, len(x_log) - 2)
        chi2_red = float((resid ** 2).sum() / dof)
        return float(slope), float(intercept), float(r_squared), slope_err, intercept_err, chi2_red

    def power_fit(
        self, x_data: List[float], y_data: List[float], y_err: Optional[List[float]] = None
    ) -> Tuple[float, float, float, float, float, float]:
        x = np.array(x_data)
        y = np.array(y_data)

        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("Power-law fit requires all x and y values to be > 0")

        xx = np.log(x)
        yy = np.log(y)

        use_weights = y_err is not None and len(y_err) == len(y) and np.all(np.array(y_err) > 0)
        if use_weights:
            sigma_y = np.array(y_err) / y
            if np.any(sigma_y <= 0) or np.any(~np.isfinite(sigma_y)):
                use_weights = False

        if use_weights:
            w = 1.0 / (sigma_y ** 2)
            s = w.sum()
            sx = (w * xx).sum()
            sy = (w * yy).sum()
            sxx = (w * xx * xx).sum()
            sxy = (w * xx * yy).sum()
            delta = s * sxx - sx * sx

            k = (s * sxy - sx * sy) / delta
            ln_c = (sy - k * sx) / s

            y_hat = k * xx + ln_c
            y_bar_w = sy / s
            sse = (w * (yy - y_hat) ** 2).sum()
            sst = (w * (yy - y_bar_w) ** 2).sum()
            r_squared_w = 1.0 - (sse / sst if sst > 0 else 0.0)

            dof = max(1, len(xx) - 2)
            chi2_red = sse / dof
            var_k = s / delta
            var_ln_c = sxx / delta
            k_err = float(np.sqrt(var_k * chi2_red))
            ln_c_err = float(np.sqrt(var_ln_c * chi2_red))
            c = float(np.exp(ln_c))
            c_err = float(c * ln_c_err)
            return float(k), c, float(r_squared_w), k_err, c_err, float(chi2_red)

        (k, ln_c), cov = np.polyfit(xx, yy, 1, cov=True)
        k_err = float(np.sqrt(cov[0, 0])) if cov.size else 0.0
        ln_c_err = float(np.sqrt(cov[1, 1])) if cov.size else 0.0
        correlation_matrix = np.corrcoef(xx, yy)
        r_squared = float(correlation_matrix[0, 1] ** 2)

        y_hat = k * xx + ln_c
        resid = yy - y_hat
        dof = max(1, len(xx) - 2)
        chi2_red = float((resid ** 2).sum() / dof)
        c = float(np.exp(ln_c))
        c_err = float(c * ln_c_err)
        return float(k), c, float(r_squared), k_err, c_err, chi2_red

    def fourier_transform(self, data: List[float], sampling_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        signal = np.array(data)
        fft_result = np.fft.fft(signal)
        n = len(signal)
        freq = np.fft.fftfreq(n, d=1 / sampling_rate)
        positive_idx = freq >= 0
        return freq[positive_idx], np.abs(fft_result[positive_idx])

    @staticmethod
    def _plot_to_base64(save_path: Optional[str] = None) -> str:
        if save_path:
            plt.savefig(save_path)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        return img_str

    def plot_linear_fit(
        self,
        x_data: List[float],
        y_data: List[float],
        title: str = "Linear Fit",
        save_path: Optional[str] = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        x_err: Optional[List[float]] = None,
        y_err: Optional[List[float]] = None,
        slope: Optional[float] = None,
        intercept: Optional[float] = None,
        r_squared: Optional[float] = None,
        slope_err: Optional[float] = None,
        intercept_err: Optional[float] = None,
    ) -> str:
        if slope is None or intercept is None or r_squared is None:
            slope, intercept, r_squared, slope_err, intercept_err, _ = self.linear_fit(x_data, y_data)

        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = slope * x_fit + intercept

        plt.figure(figsize=(10, 6))
        if x_err is not None or y_err is not None:
            plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt="o", color="blue", ecolor="gray", capsize=3)
        else:
            plt.scatter(x_data, y_data, color="blue")

        val_m, val_u = self.format_with_uncertainty(slope, slope_err if slope_err is not None else 0)
        int_m, int_u = self.format_with_uncertainty(intercept, intercept_err if intercept_err is not None else 0)
        plt.plot(x_fit, y_fit, color="red", label=f"y=({val_m}+-{val_u})x+({int_m}+-{int_u}), R^2={r_squared:.3f}")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        return self._plot_to_base64(save_path)

    def plot_log_fit(
        self,
        x_data: List[float],
        y_data: List[float],
        title: str = "Log Fit",
        save_path: Optional[str] = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        x_err: Optional[List[float]] = None,
        y_err: Optional[List[float]] = None,
        slope: Optional[float] = None,
        intercept: Optional[float] = None,
        r_squared: Optional[float] = None,
        slope_err: Optional[float] = None,
        intercept_err: Optional[float] = None,
    ) -> str:
        if slope is None or intercept is None or r_squared is None:
            slope, intercept, r_squared, slope_err, intercept_err, _ = self.log_fit(x_data, y_data)

        x = np.array(x_data)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * np.log(x_fit) + intercept

        plt.figure(figsize=(10, 6))
        if x_err is not None or y_err is not None:
            plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt="o", color="blue", ecolor="gray", capsize=3)
        else:
            plt.scatter(x_data, y_data, color="blue")

        val_m, val_u = self.format_with_uncertainty(slope, slope_err if slope_err is not None else 0)
        int_m, int_u = self.format_with_uncertainty(intercept, intercept_err if intercept_err is not None else 0)
        plt.plot(x_fit, y_fit, color="green", label=f"y=({val_m}+-{val_u})ln(x)+({int_m}+-{int_u}), R^2={r_squared:.3f}")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        return self._plot_to_base64(save_path)

    def plot_power_fit(
        self,
        x_data: List[float],
        y_data: List[float],
        title: str = "Power-Law Fit",
        save_path: Optional[str] = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        x_err: Optional[List[float]] = None,
        y_err: Optional[List[float]] = None,
        k: Optional[float] = None,
        c: Optional[float] = None,
        r_squared: Optional[float] = None,
        k_err: Optional[float] = None,
        c_err: Optional[float] = None,
    ) -> str:
        if k is None or c is None or r_squared is None:
            k, c, r_squared, k_err, c_err, _ = self.power_fit(x_data, y_data, y_err=y_err)

        x = np.array(x_data)
        x_fit = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 100)
        y_fit = c * (x_fit ** k)

        plt.figure(figsize=(10, 6))
        if x_err is not None or y_err is not None:
            plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt="o", color="blue", ecolor="gray", capsize=3)
        else:
            plt.scatter(x_data, y_data, color="blue")

        k_m, k_u = self.format_with_uncertainty(k, k_err if k_err is not None else 0)
        c_m, c_u = self.format_with_uncertainty(c, c_err if c_err is not None else 0)
        plt.plot(x_fit, y_fit, color="purple", label=f"y=({c_m}+-{c_u})x^({k_m}+-{k_u}), R^2={r_squared:.3f}")
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        return self._plot_to_base64(save_path)

    def plot_fourier_transform(
        self,
        data: List[float],
        sampling_rate: float = 1.0,
        title: str = "Fourier Transform",
        save_path: Optional[str] = None,
    ) -> str:
        freq, magnitude = self.fourier_transform(data, sampling_rate)
        plt.figure(figsize=(10, 6))
        plt.plot(freq, magnitude)
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        return self._plot_to_base64(save_path)


class LLMProcessor:
    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.model = model or os.getenv("ALIBABA_CLOUD_MODEL", "qwen-flash")
        if not self.api_key:
            raise ValueError("Please set DASHSCOPE_API_KEY")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.system_prompt = (
            "You are a physics lab report assistant. Follow: no fabrication, ask for missing key info first, keep units consistent."
        )

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None) -> str:
        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def generate_text_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[list] = None,
        enable_thinking: bool = False,
    ):
        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 2000,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if enable_thinking and ("plus" in self.model or "max" in self.model):
            params["extra_body"] = {"enable_thinking": True}

        response = self.client.chat.completions.create(**params)
        for chunk in response:
            if getattr(chunk, "choices", None):
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    yield {"type": "thinking", "text": delta.reasoning_content}
                if hasattr(delta, "content") and delta.content:
                    yield {"type": "content", "text": delta.content}
            elif getattr(chunk, "usage", None):
                usage = chunk.usage
                yield {
                    "type": "usage",
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    },
                }

    def smart_uncertainty_conversation(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        current_measurements: Dict[str, Dict[str, float]],
        enable_thinking: bool = False,
    ):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "normalize_and_compute_uncertainty",
                    "description": "Normalize a formula and compute uncertainty propagation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "formula_description": {"type": "string"},
                            "normalized_formula": {"type": "string"},
                            "measurements": {"type": "object"},
                        },
                        "required": ["formula_description", "normalized_formula", "measurements"],
                    },
                },
            }
        ]

        messages = [{"role": "system", "content": "You are a physics uncertainty-analysis assistant."}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        request_params = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.3,
            "stream": True,
        }
        if enable_thinking and "plus" in self.model:
            request_params["extra_body"] = {"enable_thinking": True}

        response = self.client.chat.completions.create(**request_params)

        content_buffer = ""
        tool_calls_buffer = []
        current_tool_call = None

        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield {"type": "thinking", "text": delta.reasoning_content}
            if hasattr(delta, "content") and delta.content:
                content_buffer += delta.content
                yield {"type": "content", "text": delta.content}

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    if tc_delta.index is not None:
                        if current_tool_call and current_tool_call["index"] != tc_delta.index:
                            tool_calls_buffer.append(current_tool_call)
                        if not current_tool_call or current_tool_call["index"] != tc_delta.index:
                            current_tool_call = {
                                "index": tc_delta.index,
                                "id": tc_delta.id or "",
                                "function": {
                                    "name": tc_delta.function.name if tc_delta.function else "",
                                    "arguments": "",
                                },
                            }
                        if tc_delta.function and tc_delta.function.arguments:
                            current_tool_call["function"]["arguments"] += tc_delta.function.arguments

        if current_tool_call:
            tool_calls_buffer.append(current_tool_call)

        if not tool_calls_buffer:
            return

        for tool_call in tool_calls_buffer:
            if tool_call["function"]["name"] != "normalize_and_compute_uncertainty":
                continue

            yield {"type": "tool_call", "tool_name": "normalize_and_compute_uncertainty"}
            args = json.loads(tool_call["function"]["arguments"])
            normalized = args["normalized_formula"]
            meas = args.get("measurements", current_measurements)

            calc = UncertaintyCalculator()
            calc.parse_formula(normalized)
            analytic = calc.compute_uncertainty_propagation_analytical(meas)

            result = {
                "success": True,
                "raw_formula": args["formula_description"],
                "normalized_formula": normalized,
                "result": analytic["result"],
                "uncertainty_total": analytic["uncertainty_total"],
                "uncertainty_a": analytic["uncertainty_a"],
                "uncertainty_b": analytic["uncertainty_b"],
                "relative_uncertainty": analytic["relative_uncertainty"],
                "contributions": analytic["contributions"],
                "contributions_detailed": analytic["contributions_detailed"],
                "partial_derivatives": analytic["partial_derivatives"],
                "measurements": meas,
            }
            yield {"type": "calculation_result", "result": result}

    def compute_uncertainty_with_llm(
        self, formula_raw: str, measurements: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        try:
            calc = UncertaintyCalculator()

            normalized = formula_raw.strip().replace("×", "*").replace("·", "*").replace("^", "**")
            if "=" in normalized:
                normalized = normalized.split("=")[-1].strip()

            calc.parse_formula(normalized)
            missing_vars = [v for v in calc.variables.keys() if v not in measurements]
            if missing_vars:
                return {
                    "success": False,
                    "error": f"Missing variables: {', '.join(missing_vars)}",
                    "raw_formula": formula_raw,
                    "normalized_formula": normalized,
                }

            analytic = calc.compute_uncertainty_propagation_analytical(measurements)
            summary_prompt = (
                "Summarize the following uncertainty result in <=90 Chinese chars, include value +- uncertainty, relative uncertainty and top contributor.\n"
                + json.dumps(
                    {
                        "result": analytic["result"],
                        "uncertainty_total": analytic["uncertainty_total"],
                        "relative_uncertainty": analytic["relative_uncertainty"],
                        "contributions": analytic["contributions"],
                    },
                    ensure_ascii=False,
                )
            )
            summary = self.generate_text(summary_prompt)

            return {
                "success": True,
                "raw_formula": formula_raw,
                "normalized_formula": normalized,
                "result": analytic["result"],
                "uncertainty_total": analytic["uncertainty_total"],
                "uncertainty_a": analytic["uncertainty_a"],
                "uncertainty_b": analytic["uncertainty_b"],
                "relative_uncertainty": analytic["relative_uncertainty"],
                "contributions": analytic["contributions"],
                "contributions_detailed": analytic["contributions_detailed"],
                "partial_derivatives": analytic["partial_derivatives"],
                "summary": summary,
                "measurements": measurements,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_formula": formula_raw,
                "normalized_formula": "",
            }

    def extract_table_from_image_bytes(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> pd.DataFrame:
        try:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            response = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Extract the table in this physics record image and return strict CSV only. "
                                    "Keep first line as header and preserve numeric precision."
                                ),
                            },
                        ],
                    }
                ],
                temperature=0.1,
            )
            csv_text = response.choices[0].message.content
            csv_text = csv_text.replace("```csv", "").replace("```", "").strip()
            return pd.read_csv(StringIO(csv_text))
        except Exception as e:
            return pd.DataFrame({"error": [f"OCR failed: {str(e)}"]})
