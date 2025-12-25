"""
符号数学计算 MCP 服务器
用于处理不确定度的符号微分和计算
"""

import json
import sympy as sp
import numpy as np
from typing import Any, Dict


class SymbolicMathTools:
    """符号数学工具集"""
    
    @staticmethod
    def compute_partial_derivative(expression: str, variable: str) -> Dict[str, str]:
        """
        计算偏导数
        
        Args:
            expression: 数学表达式字符串
            variable: 求导变量
            
        Returns:
            {
                "success": True/False,
                "result": 偏导数表达式,
                "latex": LaTeX格式,
                "simplified": 化简后的表达式
            }
        """
        try:
            # 规范化表达式
            expression = expression.replace('^', '**').replace('ln', 'log')
            
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            
            derivative = sp.diff(expr, var)
            simplified = sp.simplify(derivative)
            
            return {
                "success": True,
                "derivative": str(derivative),
                "derivative_latex": sp.latex(derivative),
                "simplified": str(simplified),
                "simplified_latex": sp.latex(simplified)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def evaluate_expression(expression: str, substitutions: Dict[str, float]) -> Dict[str, Any]:
        """
        计算表达式的值
        
        Args:
            expression: 数学表达式
            substitutions: 变量替换 {var_name: value}
            
        Returns:
            {
                "success": True/False,
                "value": 计算结果,
                "expression": 原表达式
            }
        """
        try:
            expression = expression.replace('^', '**').replace('ln', 'log')
            expr = sp.sympify(expression)
            
            # 构建替换字典
            subs_dict = {sp.Symbol(k): v for k, v in substitutions.items()}
            
            result = float(expr.subs(subs_dict))
            
            return {
                "success": True,
                "value": result,
                "expression": str(expr)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def extract_variables(expression: str) -> Dict[str, str]:
        """
        从表达式中提取所有变量
        
        Args:
            expression: 数学表达式
            
        Returns:
            {variable_name: "symbol"}
        """
        try:
            expression = expression.replace('^', '**').replace('ln', 'log')
            expr = sp.sympify(expression)
            
            variables = {str(sym): "symbol" for sym in expr.free_symbols}
            
            return {
                "success": True,
                "variables": variables,
                "count": len(variables)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def simplify_expression(expression: str) -> Dict[str, str]:
        """
        化简表达式
        
        Args:
            expression: 数学表达式
            
        Returns:
            {
                "success": True/False,
                "original": 原表达式,
                "simplified": 化简后的表达式,
                "latex": LaTeX格式
            }
        """
        try:
            expression = expression.replace('^', '**').replace('ln', 'log')
            expr = sp.sympify(expression)
            
            simplified = sp.simplify(expr)
            
            return {
                "success": True,
                "original": str(expr),
                "simplified": str(simplified),
                "simplified_latex": sp.latex(simplified)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def analyze_formula_for_uncertainties(expression: str) -> Dict[str, Any]:
        """
        分析公式结构用于误差分析
        返回每个变量的影响分析
        
        Args:
            expression: 数学表达式
            
        Returns:
            {
                "success": True/False,
                "formula": 公式,
                "variables": 变量列表,
                "linear_terms": 线性项,
                "nonlinear_terms": 非线性项,
                "recommendations": 建议
            }
        """
        try:
            expression = expression.replace('^', '**').replace('ln', 'log')
            expr = sp.sympify(expression)
            
            variables = list(expr.free_symbols)
            
            # 分析各变量的次数
            var_powers = {}
            for var in variables:
                degree = sp.degree(expr, var)
                var_powers[str(var)] = degree
            
            # 检查是否为线性
            is_linear = all(d <= 1 for d in var_powers.values())
            
            # 生成建议
            recommendations = []
            if is_linear:
                recommendations.append("该公式是线性的，误差传播直接相加")
            else:
                recommendations.append("该公式非线性，高阶项可能显著影响结果")
                for var, power in var_powers.items():
                    if power > 1:
                        recommendations.append(
                            f"变量 {var} 的幂次为 {power}，"
                            f"其不确定度的影响被放大 {power} 倍"
                        )
            
            return {
                "success": True,
                "formula": str(expr),
                "variables": [str(v) for v in variables],
                "variable_powers": {str(v): d for v, d in zip(variables, var_powers.values())},
                "is_linear": is_linear,
                "recommendations": recommendations
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# 工具列表（供MCP服务器注册）
TOOLS = {
    "compute_partial_derivative": {
        "description": "计算偏导数 ∂f/∂x",
        "parameters": {
            "expression": "数学表达式（如 m*g*h）",
            "variable": "求导变量"
        }
    },
    "evaluate_expression": {
        "description": "计算表达式的数值",
        "parameters": {
            "expression": "数学表达式",
            "substitutions": "变量值字典（如 {\"m\": 0.5, \"g\": 9.8}）"
        }
    },
    "extract_variables": {
        "description": "从表达式中提取所有变量",
        "parameters": {
            "expression": "数学表达式"
        }
    },
    "simplify_expression": {
        "description": "化简数学表达式",
        "parameters": {
            "expression": "数学表达式"
        }
    },
    "analyze_formula_for_uncertainties": {
        "description": "分析公式结构，给出误差分析建议",
        "parameters": {
            "expression": "数学表达式"
        }
    }
}


def call_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    调用符号计算工具
    """
    tools = SymbolicMathTools()
    
    if tool_name == "compute_partial_derivative":
        return tools.compute_partial_derivative(
            kwargs["expression"],
            kwargs["variable"]
        )
    elif tool_name == "evaluate_expression":
        return tools.evaluate_expression(
            kwargs["expression"],
            kwargs["substitutions"]
        )
    elif tool_name == "extract_variables":
        return tools.extract_variables(kwargs["expression"])
    elif tool_name == "simplify_expression":
        return tools.simplify_expression(kwargs["expression"])
    elif tool_name == "analyze_formula_for_uncertainties":
        return tools.analyze_formula_for_uncertainties(kwargs["expression"])
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
