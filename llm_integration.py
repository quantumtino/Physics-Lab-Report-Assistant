import os
import json
import base64
import pandas as pd
from io import StringIO
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# 确保 .env 被加载（即使未通过 app.py 直接运行本模块）
load_dotenv()

class LLMProcessor:
    def __init__(self, model: Optional[str] = None):
        # 从环境变量获取API密钥
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        # 如果传入model参数，使用传入的；否则使用环境变量，最后使用默认值
        self.model = model or os.getenv("ALIBABA_CLOUD_MODEL", "qwen-flash")
        
        # 验证API密钥是否存在
        if not self.api_key:
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 系统提示词
        self.system_prompt = (
            "你是一名专业的物理实验报告助手。严格遵守：\n"
            "- 禁止编造与夸大；仅基于已提供的数据与上下文。\n"
            "- 如关键信息不足（如实验名称、测量方法、仪器与条件、结果与结论），先提出清晰的问题再继续撰写。\n"
            "- 输出内容逻辑严谨、书面化，量纲与单位一致，符合物理实验规范。\n"
        )
    
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None) -> str:
        """
        生成文本（非流式）
        :param prompt: 用户输入的提示
        :param system_prompt: 系统提示（可选）
        :param messages: 历史消息列表（可选）
        :return: 生成的文本
        """
        try:
            if messages is None:
                messages = [
                    {"role": "system", "content": system_prompt or self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"API请求失败: {e}")
            return "API请求失败，请检查网络连接和API密钥。"
    
    def generate_text_stream(self, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None, enable_thinking: bool = False):
        """
        流式生成文本
        :param prompt: 用户输入的提示
        :param system_prompt: 系统提示（可选）
        :param messages: 历史消息列表（可选）
        :param enable_thinking: 是否启用深度思考（仅对qwen-plus模型有效）
        :yield: dict 包含 "type"（"thinking"或"content"）和 "text" 键；或直接yield文本字符串
        """
        try:
            if messages is None:
                messages = [
                    {"role": "system", "content": system_prompt or self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 2000,
                "stream": True
            }
            
            # 如果启用深度思考且是plus模型，添加extra_body
            if enable_thinking and "plus" in self.model:
                request_params["extra_body"] = {"enable_thinking": True}
            
            response = self.client.chat.completions.create(**request_params)
            
            is_answering = False
            for chunk in response:
                # 处理thinking内容（深度思考）
                if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content is not None:
                    yield {
                        "type": "thinking",
                        "text": chunk.choices[0].delta.reasoning_content
                    }
                # 处理正常内容
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    yield {
                        "type": "content",
                        "text": chunk.choices[0].delta.content
                    }
        except Exception as e:
            yield {
                "type": "error",
                "text": f"\n\n❌ API请求失败: {e}"
            }
    
    def generate_lab_report(self,
                            df: pd.DataFrame,
                            analysis: Dict[str, Any],
                            figure_hint: Optional[str] = None,
                            latex_table: Optional[str] = None,
                            experiment_hint: str = "") -> str:
        """
        生成实验报告的 LaTeX 片段：
        - 先检查关键信息是否缺失，缺失则先输出问题列表。
        - 信息充足时：撰写包含表格和图像引用的 LaTeX 片段（不编造、不夸大）。
        """
        # 将 DataFrame 压缩为 CSV 文本（避免长表打印导致截断）
        csv_text = df.to_csv(index=False)

        system_prompt = (
            "你是一名专业的物理实验报告助手。严格遵守：\n"
            "- 禁止编造与夸大；仅基于已提供的数据与上下文。\n"
            "- 如关键信息不足（如实验名称、测量方法、仪器与条件、结果与结论），先提出清晰的问题（以列表形式给出）并停止。\n"
            "- 若信息充分，再撰写实验报告片段，插入给定的表格与图像（引用即可）。\n"
            "- 量纲与单位一致，参数附带不确定度（若提供），给出必要的拟合评价（R^2、Reduced chi^2）。\n"
            "- 输出只包含 LaTeX 片段，使用中文，置于 ```latex 代码块中。\n"
        )

        figure_name = figure_hint or "实验图"
        analysis_json = json.dumps(analysis, ensure_ascii=False)

        user_prompt = f"""
已知/上下文：
- 实验提示: {experiment_hint}
- 数据表（CSV）：
{csv_text}

- 拟合/分析结果（JSON）：
{analysis_json}

- 表格 LaTeX（可直接引用）：
{latex_table if latex_table else '(未提供表格LaTeX，可省略)'}

图像：当前绘图已在应用中生成，请在 LaTeX 中以 figure 环境引用占位文件名，如 \\includegraphics[width=0.8\\linewidth]{{{{plot.png}}}}，图题为"{figure_name}"。

任务：
1) 若信息不全，先列出需要补充的关键信息问题（使用项目符号列表），不要编写正文；
2) 若信息充分，请输出“一个可直接粘贴”的 LaTeX 片段，建议含以下结构：
   - \\subsection*{{实验名称}}（若未知则留空并以注释提示）
   - \\subsubsection*{{实验数据与处理}}：插入给定表格 LaTeX（若缺失可省略）；
   - \\subsubsection*{{结果与拟合}}：说明所用拟合模型与参数（含不确定度）、R^2、Reduced $\\chi^2$；
   - 插入 figure 环境引用图像文件 plot.png（或你建议的占位名），配合合适的图题；
   - \\subsubsection*{{结论}}：给出客观、不夸大的结论；
   - 若存在局限性或可疑数据，简要说明。

注意：不要杜撰未提供的实验细节或数据；若出现模糊之处，以“待确认”标注并避免下结论。
"""

        return self.generate_text(user_prompt, system_prompt=system_prompt)

    def generate_collab_response(self,
                                 df: Optional[pd.DataFrame],
                                 analysis: Optional[Dict[str, Any]],
                                 mode: str = "plan",
                                 act_type: Optional[str] = None,
                                 user_notes: str = "") -> str:
        """
        协作式响应：支持 plan/act 两种模式。
        - plan: 输出澄清问题与行动计划（禁止编造）。
        - act: 根据 act_type 输出对应片段：
            - text: 结果与讨论（文字分析）
            - latex_table: 表格 LaTeX 片段（以代码块输出）
            - latex_figure: 插图引用 LaTeX 片段（figure 环境，引用 plot.png）
        """
        system_prompt = (
            "你是物理实验报告协作助手。\n"
            "- 严禁编造与夸大，仅基于提供的数据与结果。\n"
            "- 优先澄清需求与缺失信息，再执行输出。\n"
            "- 输出简洁凝练，术语规范，单位与量纲一致。\n"
        )

        csv_text = df.to_csv(index=False) if (df is not None and not df.empty) else "(无表格数据)"
        analysis_json = json.dumps(analysis, ensure_ascii=False) if analysis else "(无分析结果)"

        mode = (mode or "plan").lower()
        act_type = (act_type or "").lower()

        if mode == "plan":
            prompt = f"""
身份与规则：你是物理实验报告协作助手（禁止编造）。
已知表格（CSV）：
{csv_text}

已知拟合/分析结果（JSON）：
{analysis_json}

用户补充/备注：
{user_notes}

请输出：
1) 需要向用户确认/补充的关键问题（要点列表，简短）；
2) 一个执行计划（Plan）：分步骤说明接下来如何生成报告内容；
仅输出上述两部分，不要生成报告正文。
"""
            return self.generate_text(prompt, system_prompt=system_prompt)

        # act 模式
        if act_type == "text":
            prompt = f"""
身份与规则：你是物理实验报告协作助手（禁止编造）。
输入表格（CSV）：
{csv_text}

输入拟合/分析结果（JSON）：
{analysis_json}

请输出“结果与讨论”文字分析（中文，约200-400字），包含：模型与参数（及不确定度）、拟合优劣（R^2、Reduced chi^2）、异常与局限、不夸大结论。避免赘述，条理清晰。
仅输出正文，不要代码块。
"""
            return self.generate_text(prompt, system_prompt=system_prompt)

        if act_type == "latex_table":
            prompt = f"""
身份与规则：你是物理实验报告协作助手（禁止编造）。
输入表格（CSV）：
{csv_text}

请将该表格转换为 LaTeX 表格片段（table+tabular 环境，含 caption 与 label，占位名可为“实验数据表”和“tab:data”）。
仅输出 ```latex 代码块，代码块内是完整的 LaTeX 表格片段。
"""
            return self.generate_text(prompt, system_prompt=system_prompt)

        if act_type == "latex_figure":
            figure_caption = (analysis or {}).get("figure_hint", "实验图")
            prompt = f"""
身份与规则：你是物理实验报告协作助手（禁止编造）。
输入拟合/分析结果（JSON）：
{analysis_json}

请生成引用图片的 LaTeX 片段（figure 环境），引用文件名 plot.png，宽度 0.8\linewidth，图题为“{figure_caption}”。
仅输出 ```latex 代码块，代码块内是完整的 figure 片段。
"""
            return self.generate_text(prompt, system_prompt=system_prompt)

        # 默认兜底：给出可用的 act 类型提示
        prompt = "请指定 act_type 为 text、latex_table 或 latex_figure。"
        return self.generate_text(prompt, system_prompt=system_prompt)
    
    def chat_stream(self, user_message: str, history: list, context: Optional[Dict[str, Any]] = None, enable_thinking: bool = False):
        """
        流式对话，支持历史记录和实验上下文（PLAN 阶段）
        - 保持对话简洁，若信息充足则建议进入 act 阶段
        :param user_message: 用户消息
        :param history: 对话历史 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        :param context: 实验上下文（包含 dataframe, analysis_payload 等）
        :param enable_thinking: 是否启用深度思考（仅对qwen-plus模型有效）
        :yield: dict 包含 "type" 和 "text" 的字典
        """
        # 简化的 plan 阶段系统提示词
        plan_system_prompt = (
            "你是物理实验报告助手。在这个阶段，用户正在【规划】报告内容。\n"
            "- 保持简洁：3-5 句话以内回答即可。\n"
            "- 提出关键问题：如果缺少实验名称、测量方法或结果分析，列出 2-3 个必要信息。\n"
            "- 推荐行动：如果信息基本充足，建议用户进入【撰写】(act) 阶段生成报告片段。\n"
            "- 禁止编造：严格基于提供的数据和上下文。\n"
        )
        
        # 构建消息列表
        messages = [{"role": "system", "content": plan_system_prompt}]
        
        # 如果有实验上下文且是第一轮对话，在消息中注入
        if context and len(history) == 0:
            context_info = ""
            if context.get("dataframe") is not None:
                df = context["dataframe"]
                context_info += f"已有数据：{len(df)} 行数据\n"
            if context.get("analysis_payload"):
                analysis = context["analysis_payload"]
                atype = analysis.get('type', '未知')
                context_info += f"已有分析：{atype} 拟合完成\n"
            
            if context_info:
                user_message = f"【现有资源】\n{context_info}\n【用户问题】\n{user_message}"
        
        # 添加历史消息
        messages.extend(history)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 流式生成
        yield from self.generate_text_stream(prompt="", messages=messages, enable_thinking=enable_thinking)
    
    def generate_act_response(self, df: pd.DataFrame, analysis_payload: Dict[str, Any], history: list, modification: str = "", enable_thinking: bool = False):
        """
        生成报告片段（ACT 阶段）- 流式输出
        - 使用对话历史作为背景
        - 提示表格/图像的命名规范
        - 强调输出是报告片段，不是完整文档
        
        :param df: 数据表格
        :param analysis_payload: 分析结果
        :param history: 对话历史（plan 阶段的讨论）
        :param modification: 用户的修改建议
        :param enable_thinking: 是否启用深度思考（仅对qwen-plus模型有效）
        :yield: dict 包含 "type" 和 "text" 的字典
        """
        csv_text = df.to_csv(index=False) if (df is not None and not df.empty) else "(无表格数据)"
        analysis_json = json.dumps(analysis_payload, ensure_ascii=False) if analysis_payload else "(无分析结果)"
        
        # act 阶段的系统提示词
        act_system_prompt = (
            "你是物理实验报告助手。在这个阶段，用户正在【撰写】报告片段。\n"
            "重点要求：\n"
            "1. **输出是片段，不是完整报告**：只写 section/subsection 级别的内容，无需 documentclass 或完整结构。\n"
            "2. **图表命名规范**：\n"
            "   - 图表文件名格式：`<拟合类型>_<时间戳>.png`（如 `线性拟合_20250101-120000.png`）\n"
            "   - 在 LaTeX 中用 `\\includegraphics{<文件名>}` 引用。\n"
            "3. **表格数据**：使用 `tabular` 环境，含合理的表头和标题。\n"
            "4. **结果说明**：附带参数、不确定度、拟合质量指标（R²、reduced χ²）。\n"
            "5. **禁止编造**：仅基于提供的数据，出现不确定处用【待确认】标注。\n"
        )
        
        # 构建用户提示词
        act_prompt = f"""【背景】
根据之前的沟通（plan 阶段），现在需要撰写实验报告的一个小片段。

【可用数据】
数据表格（CSV）：
```csv
{csv_text}
```

分析结果（JSON）：
{analysis_json}

【用户修改/补充】
{modification if modification else "（无额外修改）"}

【任务】
生成一个 LaTeX 代码片段（含必要的 section、表格、图表引用和结果说明），约 300-500 字，满足以下条件：
- 使用 ```latex 代码块包裹输出
- 仅输出片段（无完整文档结构）
- 如有图表，使用规范命名（如 `线性拟合_<时间戳>.png`）
- 包含数据表、拟合参数、拟合质量指标
- 不编造或推断超出数据范围的内容
"""
        
        # 构建消息列表（融合 plan 阶段的历史）
        messages = [{"role": "system", "content": act_system_prompt}]
        messages.extend(history)  # 添加 plan 阶段的对话历史
        messages.append({"role": "user", "content": act_prompt})
        
        # 流式生成
        yield from self.generate_text_stream(prompt="", messages=messages, enable_thinking=enable_thinking)
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        使用多模态视觉模型从图像中提取文本
        :param image_path: 图像路径
        :return: 提取的文本内容
        """
        try:
            # 将图像转为 base64
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # 调用 qwen-vl-max 进行视觉识别
            response = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            },
                            {
                                "type": "text",
                                "text": "请提取这张图片中的所有文字内容，包括手写和打印文字。保持原有的格式和顺序。"
                            }
                        ]
                    }
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"AI视觉识别失败: {e}")
            return f"AI视觉识别失败: {str(e)}"
    
    def extract_table_from_image(self, image_path: str) -> pd.DataFrame:
        """
        使用多模态视觉模型从图像中提取表格数据
        :param image_path: 图像路径
        :return: 提取的表格 DataFrame
        """
        try:
            # 将图像转为 base64
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # 调用 qwen-vl-max 进行表格识别
            response = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            },
                            {
                                "type": "text",
                                "text": """请从这张物理实验记录图片中提取表格数据。
要求：
1. 识别所有手写的数字、中文和英文
2. 按表格结构组织数据
3. 输出为标准 CSV 格式（用逗号分隔）
4. 第一行必须是表头
5. 数值保持原有精度
6. 如果有单位，保留在表头中

示例输出格式：
序号,电压(V),电流(mA),电阻(Ω)
1,1.5,10.2,147
2,3.0,20.5,146

只输出 CSV 数据，不要添加任何解释文字。"""
                            }
                        ]
                    }
                ],
                temperature=0.1
            )
            
            # 解析返回的 CSV 文本为 DataFrame
            csv_text = response.choices[0].message.content
            
            # 清理可能的代码块标记
            csv_text = csv_text.replace('```csv', '').replace('```', '').strip()
            
            # 转换为 DataFrame
            df = pd.read_csv(StringIO(csv_text))
            return df
            
        except Exception as e:
            print(f"AI表格识别失败: {e}")
            # 返回空 DataFrame 但提供错误信息
            return pd.DataFrame({"错误": [f"识别失败: {str(e)}"]})
