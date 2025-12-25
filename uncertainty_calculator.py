"""
不确定度计算引擎
支持偏导法进行误差传播
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import re


class UncertaintyCalculator:
    """
    基于符号计算的误差传播引擎
    """
    
    def __init__(self):
        self.symbols_cache = {}
        self.formula = None
        self.variables = {}
    
    def parse_formula(self, formula_str: str) -> sp.Expr:
        """
        解析用户输入的公式字符串
        例: "m * g * h" → sympy表达式
        
        Args:
            formula_str: 数学表达式字符串
            
        Returns:
            sympy表达式对象
        """
        try:
            # 规范化表达式
            formula_str = formula_str.replace('^', '**')
            formula_str = formula_str.replace('ln', 'log')
            
            # 创建表达式
            expr = sp.sympify(formula_str)
            self.formula = expr
            
            # 自动提取变量
            self.variables = {str(sym): None for sym in expr.free_symbols}
            
            return expr
        except Exception as e:
            raise ValueError(f"公式解析失败: {str(e)}")
    
    def set_measurements(self, measurements: Dict[str, Dict[str, float]]) -> None:
        """
        设置测量数据
        
        Args:
            measurements: {
                "m": {"value": 0.5, "unit": "kg", "a_uncertainty": 0.001, "b_uncertainty": 0.0005},
                "v": {"value": 2.3, "unit": "m/s", "a_uncertainty": 0.05, "b_uncertainty": 0.02}
            }
        """
        self.variables = measurements
    
    def compute_partial_derivatives(self) -> Dict[str, str]:
        """
        计算公式中所有变量的偏导数
        
        Returns:
            {变量名: 偏导数表达式的字符串}
        """
        if self.formula is None:
            raise ValueError("请先解析公式")
        
        partial_derivs = {}
        for var_str in self.variables.keys():
            var = sp.Symbol(var_str)
            partial = sp.diff(self.formula, var)
            partial_derivs[var_str] = {
                "expression": str(partial),
                "latex": sp.latex(partial),
                "simplified": str(sp.simplify(partial))
            }
        
        return partial_derivs
    
    def compute_uncertainty_propagation_analytical(
        self,
        measurements: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        使用偏导法进行误差传播计算
        
        Args:
            measurements: 测量数据（见set_measurements）
            
        Returns:
            {
                "result": 最终值,
                "uncertainty_total": 总不确定度,
                "uncertainty_a": A类不确定度,
                "uncertainty_b": B类不确定度,
                "relative_uncertainty": 相对不确定度,
                "partial_derivatives": {变量: 偏导数值},
                "contributions": {变量: 贡献百分比},
                "latex_derivation": LaTeX推导步骤,
                "contributions_detailed": {变量: {a_contrib, b_contrib, total_contrib}}
            }
        """
        if self.formula is None:
            raise ValueError("请先解析公式")
        
        # 1. 计算偏导数（符号）
        partial_derivs_sym = {}
        for var_str in measurements.keys():
            var = sp.Symbol(var_str)
            partial_derivs_sym[var] = sp.diff(self.formula, var)
        
        # 2. 构建代入字典
        subs_dict = {}
        for var_str, data in measurements.items():
            subs_dict[sp.Symbol(var_str)] = data["value"]
        
        # 3. 计算偏导数的数值
        partial_values = {}
        for var, partial in partial_derivs_sym.items():
            try:
                partial_values[str(var)] = float(partial.subs(subs_dict))
            except:
                partial_values[str(var)] = 0.0
        
        # 4. 计算合成不确定度
        # u_total² = Σ[(∂f/∂xi)² * u(xi)²]
        uncertainty_sq_a = 0  # A类平方和
        uncertainty_sq_b = 0  # B类平方和
        contributions = {}
        contributions_detailed = {}
        
        for var_str, data in measurements.items():
            partial_val = partial_values[var_str]
            
            # A类不确定度贡献
            u_a = data.get("a_uncertainty", 0)
            contrib_a = (partial_val * u_a) ** 2
            uncertainty_sq_a += contrib_a
            
            # B类不确定度贡献
            u_b = data.get("b_uncertainty", 0)
            contrib_b = (partial_val * u_b) ** 2
            uncertainty_sq_b += contrib_b
            
            # 合成
            contrib_total = contrib_a + contrib_b
            
            contributions_detailed[var_str] = {
                "a_contribution": float(contrib_a),
                "b_contribution": float(contrib_b),
                "total_contribution": float(contrib_total),
                "partial_derivative": float(partial_val),
                "a_uncertainty": float(u_a),
                "b_uncertainty": float(u_b)
            }
        
        # 5. 计算总不确定度
        uncertainty_a = float(np.sqrt(uncertainty_sq_a))
        uncertainty_b = float(np.sqrt(uncertainty_sq_b))
        uncertainty_total = float(np.sqrt(uncertainty_sq_a + uncertainty_sq_b))
        
        # 6. 计算贡献百分比
        total_sq = uncertainty_sq_a + uncertainty_sq_b
        for var_str in measurements.keys():
            if total_sq > 0:
                contributions[var_str] = (
                    contributions_detailed[var_str]["total_contribution"] / total_sq * 100
                )
            else:
                contributions[var_str] = 0.0
        
        # 7. 计算最终结果
        result = float(self.formula.subs(subs_dict))
        
        # 8. 生成LaTeX推导
        latex_derivation = self._generate_latex_derivation(
            partial_derivs_sym,
            measurements,
            partial_values,
            contributions_detailed
        )
        
        return {
            "method": "analytical",
            "result": result,
            "uncertainty_a": uncertainty_a,
            "uncertainty_b": uncertainty_b,
            "uncertainty_total": uncertainty_total,
            "relative_uncertainty": uncertainty_total / result if result != 0 else float('inf'),
            "partial_derivatives": {
                str(var): {
                    "expression": str(partial),
                    "latex": sp.latex(partial),
                    "value": partial_values[str(var)]
                }
                for var, partial in partial_derivs_sym.items()
            },
            "contributions": contributions,
            "contributions_detailed": contributions_detailed,
            "latex_derivation": latex_derivation,
            "formula": str(self.formula),
            "formula_latex": sp.latex(self.formula)
        }
    
    def _generate_latex_derivation(
        self,
        partial_derivs: Dict,
        measurements: Dict,
        partial_values: Dict,
        contributions_detailed: Dict
    ) -> str:
        """
        生成完整的LaTeX推导过程
        """
        latex_parts = []
        
        # 1. 原始公式
        latex_parts.append("\\section*{误差分析}\n")
        latex_parts.append(f"\\textbf{{实验公式:}} $y = {sp.latex(self.formula)}$\n\n")
        
        # 2. 测量数据
        latex_parts.append("\\textbf{测量数据:}\n\\begin{align*}\n")
        for var_str, data in measurements.items():
            u_total = np.sqrt(data.get("a_uncertainty", 0)**2 + data.get("b_uncertainty", 0)**2)
            latex_parts.append(
                f"{var_str} &= ({data['value']:.4g} \\pm {u_total:.4g}) \\ {data['unit']} \\\\\n"
            )
        latex_parts.append("\\end{align*}\n\n")
        
        # 3. 偏导数
        latex_parts.append("\\textbf{偏导数计算:}\n\\begin{align*}\n")
        for var, partial in partial_derivs.items():
            var_str = str(var)
            latex_parts.append(
                f"\\frac{{\\partial y}}{{\\partial {var_str}}} &= {sp.latex(partial)} "
                f"= {partial_values[var_str]:.6g} \\\\\n"
            )
        latex_parts.append("\\end{align*}\n\n")
        
        # 4. 不确定度传播公式
        latex_parts.append("\\textbf{不确定度传播:}\n")
        latex_parts.append("$$u(y)^2 = \\sum_i \\left(\\frac{\\partial y}{\\partial x_i}\\right)^2 u(x_i)^2$$\n\n")
        
        # 5. 各项贡献
        latex_parts.append("\\textbf{各变量贡献:}\n\\begin{align*}\n")
        for var_str, contrib in contributions_detailed.items():
            latex_parts.append(
                f"u_{{{var_str}}}(y)^2 &= "
                f"({contrib['partial_derivative']:.6g})^2 \\times "
                f"({contrib['a_uncertainty']:.6g}^2 + {contrib['b_uncertainty']:.6g}^2) \\\\ "
                f"&= {contrib['total_contribution']:.6g} \\\\\n"
            )
        latex_parts.append("\\end{align*}\n\n")
        
        # 6. 合成不确定度
        total_contrib = sum(c["total_contribution"] for c in contributions_detailed.values())
        latex_parts.append(f"\\textbf{{合成不确定度:}} $u(y) = {np.sqrt(total_contrib):.6g}$\n\n")
        
        return "".join(latex_parts)
    

def validate_measurement_data(measurements: Dict[str, Dict[str, float]]) -> Tuple[bool, List[str]]:
    """
    验证测量数据的合理性
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    for var_str, data in measurements.items():
        value = data.get("value", 0)
        a_unc = data.get("a_uncertainty", 0)
        b_unc = data.get("b_uncertainty", 0)
        
        # 检查相对不确定度
        total_unc = np.sqrt(a_unc**2 + b_unc**2)
        if value != 0:
            rel_unc = total_unc / abs(value)
            if rel_unc > 0.5:
                errors.append(f"{var_str}: 相对不确定度过大 ({rel_unc:.1%}), 可能存在测量错误")
            elif rel_unc < 0.001:
                errors.append(f"{var_str}: 相对不确定度过小 ({rel_unc:.2%}), 可能低估了误差")
        
        # 检查不确定度为负
        if a_unc < 0 or b_unc < 0:
            errors.append(f"{var_str}: 不确定度不能为负")
        
        # 检查单位
        if "unit" not in data or not data["unit"]:
            errors.append(f"{var_str}: 未指定单位")
    
    return len(errors) == 0, errors
