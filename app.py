import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import StringIO
import os
from PIL import Image
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 导入自定义模块
from latex_generator import LatexGenerator
from analysis_module import DataAnalyzer
from llm_integration import LLMProcessor
from uncertainty_calculator import UncertaintyCalculator, validate_measurement_data

# 设置页面配置
st.set_page_config(
    page_title="物理实验报告助手",
    page_icon="🔬",
    layout="wide"
)

# 初始化会话状态
if 'ocr_result' not in st.session_state:
    st.session_state.ocr_result = None
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'latex_output' not in st.session_state:
    st.session_state.latex_output = ""
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'context_initialized' not in st.session_state:
    st.session_state.context_initialized = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "qwen-flash"  # 默认模型
if 'uncertainty_conversation' not in st.session_state:
    st.session_state.uncertainty_conversation = []
if 'uncertainty_measurements' not in st.session_state:
    st.session_state.uncertainty_measurements = {}
if 'uncertainty_formula' not in st.session_state:
    st.session_state.uncertainty_formula = None
if 'uncertainty_analysis_result' not in st.session_state:
    st.session_state.uncertainty_analysis_result = None
if 'clear_uncertainty_inputs' not in st.session_state:
    st.session_state.clear_uncertainty_inputs = False
if 'show_add_success' not in st.session_state:
    st.session_state.show_add_success = False

def main():
    st.title("🔬 物理实验报告助手")
    st.markdown("---")
    
    # 创建侧边栏导航
    st.sidebar.header("📄 功能导航")
    nav_options = ["OCR识别", "数据分析", "误差分析", "LLM协作"]
    
    # 默认显示第一页，之后记住用户选择
    current = st.session_state.get("current_page", nav_options[0])
    try:
        idx = nav_options.index(current)
    except ValueError:
        idx = 0
    
    # 显示步骤提示
    st.sidebar.divider()
    if current == "OCR识别":
        st.sidebar.info("📍 步骤 1/4\n提取实验数据表格")
    elif current == "数据分析":
        st.sidebar.info("📍 步骤 2/4\n拟合分析与可视化")
    elif current == "误差分析":
        st.sidebar.info("📍 步骤 3/4\n不确定度分析")
    elif current == "LLM协作":
        st.sidebar.info("📍 步骤 4/4\n撰写实验报告")
    st.sidebar.divider()
    
    # Radio 选择导航（用户点击时更新 current_page）
    page = st.sidebar.radio("页面选择", nav_options, index=idx, label_visibility="collapsed")
    st.session_state["current_page"] = page
    
    if page == "OCR识别":
        ocr_page()
    elif page == "数据分析":
        analysis_page()
    elif page == "误差分析":
        uncertainty_page()
    elif page == "LLM协作":
        llm_page_new()

def ocr_page():
    st.header("📸 OCR 识别")
    st.caption("使用 AI 视觉识别提取实验数据表格")
    
    # 上传图像
    uploaded_file = st.file_uploader("上传实验数据图像", type=["jpg", "jpeg", "png", "bmp"], key="ocr_image_uploader")
    
    if uploaded_file is not None:
        # 显示上传的图像
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图像", use_container_width=True)
        
        # 保存上传的图像到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            image_path = tmp_file.name
        
        # 使用 AI 视觉识别
        try:
            llm_processor = LLMProcessor(model=st.session_state.selected_model)
            
            # 提取表格
            if st.button("📋 提取数据表格"):
                with st.spinner("AI 正在识别表格数据..."):
                    df = llm_processor.extract_table_from_image(image_path)
                    
                    if not df.empty and "错误" not in df.columns:
                        # 保存原始识别结果到 session_state
                        st.session_state.ocr_dataframe = df
                        st.success("✅ 表格识别完成！")
                    else:
                        st.error("表格识别失败，请尝试：\n1. 确保图像清晰\n2. 表格结构明显\n3. 检查是否包含数据表格")
            
            # 显示识别的表格（如果有的话）
            if "ocr_dataframe" in st.session_state and not st.session_state.ocr_dataframe.empty:
                df = st.session_state.ocr_dataframe
                
                st.subheader("📊 识别的表格")
                st.dataframe(df)
                
                # 允许用户编辑
                st.info("💡 请在下方编辑表格数据，保存后进入拟合步骤")
                edited_df = st.data_editor(df, num_rows="dynamic", key="table_editor")
                
                # 保存DataFrame到会话状态
                if st.button("✅ 确认并保存表格", use_container_width=True):
                    st.session_state.dataframe = edited_df
                    st.session_state.ocr_dataframe = edited_df  # 同时更新
                    st.success("🎉 第一阶段完成！表格已保存！")
                    st.info("⬇️ **下一步**：点击左侧菜单 → '数据分析' 进入拟合分析阶段")
        
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.info("请在 .env 文件中配置 DASHSCOPE_API_KEY")
        except Exception as e:
            st.error(f"发生错误: {str(e)}")
        
        # 清理临时文件
        try:
            os.unlink(image_path)
        except:
            pass

def uncertainty_page():
    """
    对话式误差分析页面（简洁版）
    流程：先填测量数据与不确定度 → 输入公式 → 系统调用符号工具计算 → 简要回复
    """
    st.header("🎯 误差与不确定度分析")
    st.caption("单条录入物理量（符号、数值、单位、不确定度），系统会自动规范公式并调用符号工具计算。")

    if "uncertainty_table" not in st.session_state:
        st.session_state.uncertainty_table = []
    if "uncertainty_summary" not in st.session_state:
        st.session_state.uncertainty_summary = None
    if "clear_uncertainty_inputs" not in st.session_state:
        st.session_state.clear_uncertainty_inputs = False
    if "show_add_success" not in st.session_state:
        st.session_state.show_add_success = False

    # 表单式录入
    st.subheader("📋 逐项录入测量量")
    st.caption("一次添加一个量:符号、数值、单位、A类σ、B类σ。重复同名会覆盖。")
    
    # 使用计数器强制重置表单
    if "form_counter" not in st.session_state:
        st.session_state.form_counter = 0
    
    with st.form(f"uncertainty_form_{st.session_state.form_counter}"):
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1, 1, 1])
        var_name = c1.text_input("符号", placeholder="m, v, R", key=f"var_name_{st.session_state.form_counter}")
        var_value = c2.text_input("数值", placeholder="0.0", key=f"var_value_{st.session_state.form_counter}")
        var_unit = c3.text_input("单位", placeholder="kg, m/s", key=f"var_unit_{st.session_state.form_counter}")
        var_a = c4.text_input("A类(σ)", placeholder="0.0", key=f"var_a_{st.session_state.form_counter}")
        var_b = c5.text_input("B类(σ)", placeholder="0.0", key=f"var_b_{st.session_state.form_counter}")
        submitted = st.form_submit_button("保存/更新该变量", type="primary", use_container_width=True)
    
    # 显示成功消息（如果有）
    if st.session_state.show_add_success:
        st.success("✅ 已成功添加/更新变量！")
        st.session_state.show_add_success = False

    if submitted:
        name = var_name.strip()
        if not name:
            st.warning("请先填写变量符号")
        else:
            try:
                val = float(var_value.strip()) if var_value.strip() else 0.0
                a_val = float(var_a.strip()) if var_a.strip() else 0.0
                b_val = float(var_b.strip()) if var_b.strip() else 0.0
                
                entry = {
                    "变量": name,
                    "数值": val,
                    "单位": var_unit.strip(),
                    "A类(σ)": a_val,
                    "B类(σ)": b_val
                }
                # 覆盖同名变量
                replaced = False
                for idx, row in enumerate(st.session_state.uncertainty_table):
                    if row.get("变量", "").strip() == name:
                        st.session_state.uncertainty_table[idx] = entry
                        replaced = True
                        break
                if not replaced:
                    st.session_state.uncertainty_table.append(entry)
                st.session_state.uncertainty_analysis_result = None
                st.session_state.uncertainty_summary = None
                # 增加计数器以重置表单
                st.session_state.form_counter += 1
                st.session_state.show_add_success = True
                st.rerun()
            except ValueError:
                st.error("数值格式错误，请输入有效数字")

    # 已添加的测量量预览
    st.markdown("**当前测量量**")
    if st.session_state.uncertainty_table:
        preview_df = pd.DataFrame(st.session_state.uncertainty_table)
        st.dataframe(preview_df, use_container_width=True, height=240)

        col_del = st.columns([2, 1, 1])
        with col_del[0]:
            st.caption("A类=统计误差，B类=系统误差；单位需一致。")
        with col_del[1]:
            remove_opt = st.selectbox("删除变量", options=["无"] + [row["变量"] for row in st.session_state.uncertainty_table], key="uncertainty_remove_opt")
        with col_del[2]:
            if st.button("🗑️ 删除所选") and remove_opt != "无":
                st.session_state.uncertainty_table = [row for row in st.session_state.uncertainty_table if row.get("变量") != remove_opt]
                st.session_state.uncertainty_analysis_result = None
                st.session_state.uncertainty_summary = None
                st.rerun()

        if st.button("♻️ 清空全部数据", type="secondary"):
            st.session_state.uncertainty_table = []
            st.session_state.uncertainty_measurements = {}
            st.session_state.uncertainty_analysis_result = None
            st.session_state.uncertainty_summary = None
            st.rerun()
    else:
        st.info("暂无测量量，请用上方表单添加。")

    # 整理表格为计算所需结构（供对话使用）
    measurements = {}
    for row in st.session_state.uncertainty_table:
        name = str(row.get("变量", "")).strip()
        if not name:
            continue
        measurements[name] = {
            "value": float(row.get("数值", 0) or 0),
            "unit": row.get("单位", ""),
            "a_uncertainty": float(row.get("A类(σ)", 0) or 0),
            "b_uncertainty": float(row.get("B类(σ)", 0) or 0),
        }
    st.session_state.uncertainty_measurements = measurements

    # 聊天区（输入框在下）
    st.divider()
    st.subheader("💬 不确定度对话")
    st.caption("在下方对话中描述实验公式和测量情况，AI会引导你完成不确定度计算。")
    chat_container = st.container()
    if not st.session_state.uncertainty_conversation:
        st.session_state.uncertainty_conversation = [
            {
                "role": "assistant",
                "content": "你好！我会帮你完成不确定度分析。\n\n请告诉我：\n1. 实验的计算公式（可以用自然语言描述，如'动能等于二分之一乘以质量乘以速度平方'）\n2. 各变量的测量值、单位和不确定度（A类和B类）\n\n我会帮你规范公式并调用符号工具计算。"
            }
        ]

    with chat_container:
        for msg in st.session_state.uncertainty_conversation:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    user_msg = st.chat_input("描述公式或提问（例如：计算动能，已知质量和速度）")
    if user_msg:
        st.session_state.uncertainty_conversation.append({"role": "user", "content": user_msg})
        try:
            llm = LLMProcessor(model=st.session_state.selected_model)
            with st.chat_message("assistant", avatar="🤖"):
                ph = st.empty()
                resp = ""
                thinking_text = ""
                tool_calls_text = ""
                
                # 判断是否使用plus模型并启用深度思考
                enable_thinking = "plus" in st.session_state.selected_model
                
                # 调用智能不确定度对话（可能触发计算）
                for chunk in llm.smart_uncertainty_conversation(
                    user_msg,
                    st.session_state.uncertainty_conversation[:-1],
                    measurements,
                    enable_thinking=enable_thinking
                ):
                    if isinstance(chunk, dict):
                        chunk_type = chunk.get("type", "")
                        chunk_text = chunk.get("text", "")
                        
                        if chunk_type == "thinking":
                            thinking_text += chunk_text
                        elif chunk_type == "content":
                            resp += chunk_text
                        elif chunk_type == "tool_call":
                            # 显示MCP工具调用
                            tool_calls_text += f"\n\n🔧 **调用工具**: {chunk.get('tool_name', 'unknown')}\n"
                        elif chunk_type == "calculation_result":
                            # 保存计算结果
                            calc_result = chunk.get("result")
                            if calc_result and calc_result.get("success"):
                                st.session_state.uncertainty_analysis_result = calc_result
                                st.session_state.uncertainty_summary = calc_result.get("summary", "")
                                
                                # 显示详细计算结果
                                result_display = f"\n\n---\n\n🎯 **计算结果**\n\n"
                                result_display += f"**原始公式**: {calc_result.get('raw_formula', 'N/A')}\n\n"
                                result_display += f"**规范化公式**: `{calc_result.get('normalized_formula', 'N/A')}`\n\n"
                                
                                # 显示LaTeX公式
                                if calc_result.get('partial_derivatives'):
                                    result_display += "**偏导数**:\n\n"
                                    for var, deriv_info in calc_result['partial_derivatives'].items():
                                        latex_expr = deriv_info.get('latex', '')
                                        value = deriv_info.get('value', 0)
                                        result_display += f"- $\\frac{{\\partial f}}{{\\partial {var}}} = {latex_expr}$ ≈ {value:.4g}\n"
                                    result_display += "\n"
                                
                                # 结果与不确定度
                                result_display += f"**最终结果**: {calc_result.get('result', 0):.6g} ± {calc_result.get('uncertainty_total', 0):.4g}\n\n"
                                result_display += f"- A类不确定度: {calc_result.get('uncertainty_a', 0):.4g}\n"
                                result_display += f"- B类不确定度: {calc_result.get('uncertainty_b', 0):.4g}\n"
                                result_display += f"- 相对不确定度: {calc_result.get('relative_uncertainty', 0):.2%}\n\n"
                                
                                # 各变量贡献
                                if calc_result.get('contributions'):
                                    result_display += "**各变量贡献占比**:\n\n"
                                    sorted_contrib = sorted(calc_result['contributions'].items(), key=lambda x: x[1], reverse=True)
                                    for var, contrib in sorted_contrib:
                                        bar_length = int(contrib / 5)  # 每5%一个方块
                                        bar = "█" * bar_length
                                        result_display += f"- **{var}**: {contrib:.1f}% {bar}\n"
                                
                                resp += result_display
                    else:
                        resp += str(chunk)
                    
                    # 实时显示
                    display_parts = []
                    if thinking_text:
                        display_parts.append(f"🧠 **思考过程**\n\n```\n{thinking_text}\n```")
                    if tool_calls_text:
                        display_parts.append(tool_calls_text)
                    if resp:
                        display_parts.append(resp)
                    
                    display_text = "\n\n---\n\n".join(display_parts) + "▌"
                    ph.markdown(display_text, unsafe_allow_html=True)
                
                # 最终显示
                display_parts = []
                if thinking_text:
                    display_parts.append(f"🧠 **思考过程**\n\n```\n{thinking_text}\n```")
                if tool_calls_text:
                    display_parts.append(tool_calls_text)
                if resp:
                    display_parts.append(resp)
                
                display_text = "\n\n---\n\n".join(display_parts)
                ph.markdown(display_text, unsafe_allow_html=True)
                
                st.session_state.uncertainty_conversation.append({"role": "assistant", "content": resp})
            st.rerun()
        except Exception as e:
            st.error(f"LLM 交互失败: {str(e)}")

    # 传递结果到写作页（移到对话下方）
    st.divider()
    if st.button("📝 传给写作AI", disabled=st.session_state.uncertainty_analysis_result is None, use_container_width=True, type="primary"):
        if st.session_state.uncertainty_analysis_result:
            # 同时传递计算结果和对话历史
            st.session_state.passed_uncertainty_result = st.session_state.uncertainty_analysis_result
            st.session_state.passed_uncertainty_conversation = st.session_state.uncertainty_conversation
            st.success("✅ 已将误差分析结果和对话历史传递到写作页面！")
            # 自动切换到LLM协作页
            st.session_state["current_page"] = "LLM协作"
            st.rerun()
        else:
            st.warning("请先完成不确定度计算后再传递结果")

def analysis_page():
    st.header("📈 数据分析与拟合")
    
    # 数据来源选择
    st.subheader("📊 数据来源")
    
    # 获取可用的数据源
    has_ocr_data = st.session_state.dataframe is not None and not st.session_state.dataframe.empty
    
    # 数据源选项卡
    if has_ocr_data:
        st.info("✅ 已从 OCR 识别获得数据表")
        data_source = st.radio(
            "选择数据来源",
            ["使用 OCR 识别的表格", "上传 CSV 文件", "手动输入数据"],
            help="优先推荐使用 OCR 识别的表格"
        )
    else:
        st.warning("⚠️ 未检测到 OCR 识别的表格")
        data_source = st.radio(
            "选择数据来源",
            ["上传 CSV 文件", "手动输入数据"],
            help="请选择一种方式提供数据"
        )
    
    df = None
    
    # 数据来源 1: OCR 识别的表格
    if data_source == "使用 OCR 识别的表格" and has_ocr_data:
        df = st.session_state.dataframe
        st.subheader("📋 OCR 识别的表格")
        col1, col2 = st.columns([4, 1])
        with col1:
            st.dataframe(df, use_container_width=True)
        with col2:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📋 复制表格\n(CSV格式)",
                data=csv_data,
                file_name="data_table.csv",
                mime="text/csv",
                use_container_width=True
            )
        st.caption("💡 提示：可在下方编辑数据或选择其他数据源")
    
    # 数据来源 2: CSV 上传
    elif data_source == "上传 CSV 文件":
        st.subheader("📥 上传 CSV 文件")
        uploaded_csv = st.file_uploader("选择 CSV 文件", type="csv", key="analysis_csv_upload")
        
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.success("✅ CSV 文件加载成功")
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.dataframe(df, use_container_width=True)
                with col2:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📋 复制表格\n(CSV格式)",
                        data=csv_data,
                        file_name="data_table_export.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.info("💡 数据已加载，请点击下方按钮确认使用")
                if st.button("✅ 确认并使用此数据", use_container_width=True):
                    st.session_state.dataframe = df
                    st.success("✅ 数据已保存！")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ CSV 读取失败: {e}")
                df = None
        else:
            st.info("📂 请上传一个 CSV 文件（包含表头和数值数据）")
    
    # 数据来源 3: 手动输入
    elif data_source == "手动输入数据":
        st.subheader("✏️ 手动输入数据")
        
        # 两种手动输入方式
        input_method = st.radio("输入方式", ["表格编辑器", "粘贴 CSV 格式文本"], horizontal=True)
        
        if input_method == "表格编辑器":
            st.caption("使用下方编辑器输入数据（可添加/删除行）")
            
            # 初始化示例数据
            if "manual_data_df" not in st.session_state:
                st.session_state.manual_data_df = pd.DataFrame({
                    "列名1": [1.0, 2.0, 3.0],
                    "列名2": [2.0, 4.0, 6.0]
                })
            
            edited_df = st.data_editor(
                st.session_state.manual_data_df,
                num_rows="dynamic",
                key="manual_data_editor",
                use_container_width=True
            )
            
            st.info("💡 编辑完成后，请点击下方按钮确认使用")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 确认手动输入的数据", use_container_width=True):
                    st.session_state.dataframe = edited_df
                    st.session_state.manual_data_df = edited_df
                    st.success("🎉 数据已保存！")
                    st.info("⬇️ **下一步**：点击左侧菜单 → '数据分析' 进入拟合分析阶段")
                    st.rerun()
            with col2:
                csv_export = edited_df.to_csv(index=False)
                st.download_button(
                    "📋 导出为 CSV",
                    data=csv_export,
                    file_name="manual_data.csv",
                    mime="text/csv"
                )
            
            df = edited_df
        
        else:  # 粘贴 CSV 文本
            st.caption("粘贴 CSV 格式的文本（以逗号分隔，第一行为表头）")
            csv_text = st.text_area(
                "CSV 格式文本",
                value="列名1,列名2\n1.0,2.0\n2.0,4.0\n3.0,6.0",
                height=150,
                key="manual_csv_text"
            )
            
            if st.button("🔄 解析 CSV 文本"):
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(csv_text))
                    st.success("✅ 解析成功")
                    st.dataframe(df, use_container_width=True)
                    
                    st.info("💡 解析完成，请点击下方按钮确认使用")
                    if st.button("✅ 确认使用此数据", use_container_width=True):
                        st.session_state.dataframe = df
                        st.success("🎉 数据已保存！")
                        st.info("⬇️ **下一步**：点击左侧菜单 → '数据分析' 进入拟合分析阶段")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ 解析失败: {e}")
                    st.info("📌 确保 CSV 格式正确：第一行为列名，数据用逗号分隔")
    
    # 检查是否有可用数据
    if df is None or df.empty:
        st.warning("⚠️ 暂无可用数据，请完成上述步骤")
        return
    
    # ===== 数据预处理和分析 =====
    st.divider()
    st.subheader("⚙️ 分析参数设置")
    
    # 选择列进行分析
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("❌ 数据不包含足够的数值列（至少需要 2 列）")
        st.info("请上传包含至少 2 列数值数据的文件")
        return
    
    # 选择 X、Y 列
    col1, col2 = st.columns(2)
    x_col = col1.selectbox("选择X轴列", numeric_cols, key="x_col")
    y_col = col2.selectbox("选择Y轴列", numeric_cols, key="y_col")

    # 高级选项开关
    advanced = st.checkbox("高级选项", value=False, help="开启后可选择对数拟合、误差棒、采样率等")

    # 拟合类型
    analysis_options = ["线性拟合", "傅里叶变换"]
    if advanced:
        analysis_options.insert(1, "对数拟合 (x>0)")
        analysis_options.insert(2, "双对数拟合 (x>0, y>0)")
    analysis_type = st.selectbox("选择分析类型", analysis_options)

    # 误差列（仅拟合类使用）
    x_err = y_err = None
    if advanced and (analysis_type == "线性拟合" or analysis_type.startswith("对数拟合") or analysis_type.startswith("双对数拟合")):
        err_cols = ["无"] + numeric_cols
        err_c1, err_c2 = st.columns(2)
        x_err_col = err_c1.selectbox("X 误差列", err_cols, key="x_err_col")
        y_err_col = err_c2.selectbox("Y 误差列", err_cols, key="y_err_col")
        if x_err_col != "无":
            x_err = df[x_err_col].tolist()
        if y_err_col != "无":
            y_err = df[y_err_col].tolist()

    # 采样率（FFT 高级）
    sampling_rate = 1.0
    if analysis_type == "傅里叶变换" and advanced:
        sampling_rate = st.number_input("采样率 (Hz)", min_value=0.0001, value=1.0, step=0.1)

    # 坐标轴标签（允许自定义物理量名称）
    axis_c1, axis_c2 = st.columns(2)
    xlabel_in = axis_c1.text_input("X 轴标签", value=str(x_col) if x_col else "X")
    ylabel_in = axis_c2.text_input("Y 轴标签", value=str(y_col) if y_col else "Y")

    if x_col and y_col:
        x_data = df[x_col].tolist()
        y_data = df[y_col].tolist()

        if st.button("🔍 执行分析"):
            analyzer = DataAnalyzer()

            if analysis_type == "线性拟合":
                with st.spinner("正在执行线性拟合..."):
                    slope, intercept, r_squared, slope_err, intercept_err, chi2r = analyzer.linear_fit(
                        x_data, y_data, y_err=y_err, x_err=x_err
                    )

                    st.subheader("📈 线性拟合结果")
                    c1, c2, c3 = st.columns(3)
                    slope_str, slope_unc = analyzer.format_with_uncertainty(slope, slope_err)
                    intercept_str, intercept_unc = analyzer.format_with_uncertainty(intercept, intercept_err)
                    c1.metric("斜率", f"{slope_str} ± {slope_unc}")
                    c2.metric("截距", f"{intercept_str} ± {intercept_unc}")
                    c3.metric("加权 R²", f"{r_squared:.4f}")
                    st.caption(f"Reduced $\\chi^2$ = {chi2r:.3f}")

                    # 保存图像到本地，命名：线性拟合+时间戳
                    plots_dir = os.path.join(os.getcwd(), "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    fname = f"线性拟合_{ts}.png"
                    full_path = os.path.join(plots_dir, fname)

                    plot_data = analyzer.plot_linear_fit(
                        x_data, y_data,
                        title=f"{x_col} vs {y_col} Linear Fit",
                        xlabel=xlabel_in,
                        ylabel=ylabel_in,
                        x_err=x_err,
                        y_err=y_err,
                        slope=slope,
                        intercept=intercept,
                        r_squared=r_squared,
                        slope_err=slope_err,
                        intercept_err=intercept_err,
                        save_path=full_path
                    )

                    st.image(f"data:image/png;base64,{plot_data}", caption="线性拟合图")

                    st.session_state.analysis_result = {
                        "type": "linear_fit",
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_squared,
                        "slope_err": slope_err,
                        "intercept_err": intercept_err,
                        "plot_data": plot_data
                    }
                    st.success("✅ 分析完成！")
                    # 保存统一的分析上下文，供 LLM 协作页使用
                    st.session_state["analysis_payload"] = {
                        "type": "linear",
                        "x_col": x_col,
                        "y_col": y_col,
                        "slope": slope,
                        "intercept": intercept,
                        "slope_err": slope_err,
                        "intercept_err": intercept_err,
                        "r_squared": r_squared,
                        "reduced_chi2": chi2r,
                        "figure_hint": "线性拟合图"
                    }
                    # 保存图像文件信息供 LLM 协作页使用
                    st.session_state["plot_file_path"] = full_path
                    st.session_state["plot_file_name"] = fname
                    
                    # 第一步：突出显示下一步提示
                    st.divider()
                    st.success("🎉 第二阶段完成！线性拟合已生成！")
                    with st.container(border=True):
                        st.markdown("### 💡 下一步")
                        st.markdown("""
                        在左侧菜单选择 **'LLM协作'** 与大语言模型共同撰写实验总结
                        
                        - 📝 Plan阶段：澄清实验背景和方法
                        - ✍️ Act阶段：生成结果分析和讨论
                        """)
                    
                    # 第二步：自定义选项（可选）
                    st.divider()
                    st.subheader("🔧 自定义拟合结果（可选）")
                    
                    custom_tab1, custom_tab2 = st.tabs(["📝 输入LaTeX", "📥 导入CSV"])
                    
                    with custom_tab1:
                        st.caption("手动输入自定义 LaTeX 内容（可覆盖自动生成的结果）")
                        custom_latex = st.text_area("LaTeX代码", height=150, key="linear_latex")
                        if custom_latex:
                            st.session_state["custom_latex"] = custom_latex
                            st.info("✅ 自定义 LaTeX 已保存，将在下载时包含")
                    
                    with custom_tab2:
                        st.caption("导入 CSV 文件以更新或扩展数据")
                        csv_file = st.file_uploader("选择CSV文件", type="csv", key="linear_csv_upload")
                        if csv_file:
                            try:
                                new_df = pd.read_csv(csv_file)
                                st.dataframe(new_df)
                                if st.button("✅ 确认导入", key="linear_csv_confirm"):
                                    st.session_state.dataframe = new_df
                                    st.success("✅ 数据已更新，请重新执行分析")
                            except Exception as e:
                                st.error(f"❌ CSV 导入失败: {e}")
                    
                    # 第三步：复制按钮（作为辅助功能）
                    st.divider()
                    st.subheader("📋 复制拟合结果（可选）")
                    col1, col2 = st.columns(2)
                    with col1:
                        result_txt = f"""线性拟合结果：
斜率: {slope_str} ± {slope_unc}
截距: {intercept_str} ± {intercept_unc}
加权R²: {r_squared:.4f}
Reduced χ²: {chi2r:.3f}
图文件: {fname}"""
                        st.download_button(
                            "📋 复制拟合结果",
                            data=result_txt,
                            file_name="linear_fit_result.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col2:
                        st.caption("💡 点击下载拟合结果\n（可在其他地方使用）")

            elif analysis_type.startswith("对数拟合"):
                try:
                    with st.spinner("正在执行对数拟合..."):
                        slope, intercept, r_squared, slope_err, intercept_err, chi2r = analyzer.log_fit(
                            x_data, y_data, y_err=y_err
                        )

                        st.subheader("📈 对数拟合结果")
                        c1, c2, c3 = st.columns(3)
                        slope_str, slope_unc = analyzer.format_with_uncertainty(slope, slope_err)
                        intercept_str, intercept_unc = analyzer.format_with_uncertainty(intercept, intercept_err)
                        c1.metric("系数 a", f"{slope_str} ± {slope_unc}")
                        c2.metric("截距 b", f"{intercept_str} ± {intercept_unc}")
                        c3.metric("加权 R²", f"{r_squared:.4f}")
                        st.caption(f"Reduced $\\chi^2$ = {chi2r:.3f}")

                        # 保存图像到本地，命名：对数拟合+时间戳
                        plots_dir = os.path.join(os.getcwd(), "plots")
                        os.makedirs(plots_dir, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                        fname = f"对数拟合_{ts}.png"
                        full_path = os.path.join(plots_dir, fname)

                        plot_data = analyzer.plot_log_fit(
                            x_data, y_data,
                            title=f"{x_col} vs {y_col} Log Fit",
                            xlabel=xlabel_in,
                            ylabel=ylabel_in,
                            x_err=x_err,
                            y_err=y_err,
                            slope=slope,
                            intercept=intercept,
                            r_squared=r_squared,
                            slope_err=slope_err,
                            intercept_err=intercept_err,
                            save_path=full_path
                        )

                        st.image(f"data:image/png;base64,{plot_data}", caption="对数拟合图")

                        st.session_state.analysis_result = {
                            "type": "log_fit",
                            "a": slope,
                            "b": intercept,
                            "r_squared": r_squared,
                            "a_err": slope_err,
                            "b_err": intercept_err,
                            "plot_data": plot_data
                        }
                        st.success("✅ 分析完成！")
                        # 保存统一的分析上下文，供 LLM 协作页使用
                        st.session_state["analysis_payload"] = {
                            "type": "log",
                            "x_col": x_col,
                            "y_col": y_col,
                            "a": slope,
                            "b": intercept,
                            "a_err": slope_err,
                            "b_err": intercept_err,
                            "r_squared": r_squared,
                            "reduced_chi2": chi2r,
                            "figure_hint": "对数拟合图"
                        }
                        st.session_state["plot_file_path"] = full_path
                        st.session_state["plot_file_name"] = fname
                        
                        # 第一步：突出显示下一步提示
                        st.divider()
                        st.success("🎉 第二阶段完成！对数拟合已生成！")
                        with st.container(border=True):
                            st.markdown("### 💡 下一步")
                            st.markdown("""
                            在左侧菜单选择 **'LLM协作'** 与大语言模型共同撰写实验总结
                            
                            - 📝 Plan阶段：澄清实验背景和方法
                            - ✍️ Act阶段：生成结果分析和讨论
                            """)
                        
                        # 第二步：复制按钮（作为辅助功能）
                        st.divider()
                        st.subheader("📋 复制拟合结果（可选）")
                        col1, col2 = st.columns(2)
                        with col1:
                            result_txt = f"""对数拟合结果：
系数a: {slope_str} ± {slope_unc}
截距b: {intercept_str} ± {intercept_unc}
加权R²: {r_squared:.4f}
Reduced χ²: {chi2r:.3f}
图文件: {fname}"""
                            st.download_button(
                                "📋 复制拟合结果",
                                data=result_txt,
                                file_name="log_fit_result.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        with col2:
                            st.caption("💡 点击下载拟合结果\n（可在其他地方使用）")
                except ValueError as e:
                    st.error(f"❌ 对数拟合失败: {e}")

            elif analysis_type == "傅里叶变换":
                with st.spinner("正在执行傅里叶变换..."):
                    freq, magnitude = analyzer.fourier_transform(y_data, sampling_rate=sampling_rate)

                    st.subheader("📊 傅里叶变换结果")
                    st.write(f"频率范围: {freq[0]:.4f} - {freq[-1]:.4f}")
                    st.write(f"频谱峰值: {max(magnitude):.4f}")

                    # 保存图像到本地，命名：傅里叶变换+时间戳
                    plots_dir = os.path.join(os.getcwd(), "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    fname = f"傅里叶变换_{ts}.png"
                    full_path = os.path.join(plots_dir, fname)

                    plot_data = analyzer.plot_fourier_transform(
                        y_data,
                        sampling_rate=sampling_rate,
                        title=f"{y_col} 傅里叶变换",
                        save_path=full_path
                    )

                    st.image(f"data:image/png;base64,{plot_data}", caption="频谱图")

                    st.session_state.analysis_result = {
                        "type": "fourier_transform",
                        "plot_data": plot_data
                    }
                    
                    # 第一步：突出显示下一步提示
                    st.divider()
                    st.success("🎉 第二阶段完成！傅里叶变换已生成！")
                    with st.container(border=True):
                        st.markdown("### 💡 下一步")
                        st.markdown("""
                        在左侧菜单选择 **'LLM协作'** 与大语言模型共同撰写实验总结
                        
                        - 📝 Plan阶段：澄清实验背景和方法
                        - ✍️ Act阶段：生成结果分析和讨论
                        """)
                    
                    # 第二步：复制按钮（作为辅助功能）
                    st.divider()
                    st.subheader("📋 复制分析结果（可选）")
                    col1, col2 = st.columns(2)
                    with col1:
                        result_txt = f"""傅里叶变换分析：
图文件: {fname}
分析列: {y_col}
采样率: {sampling_rate} Hz"""
                        st.download_button(
                            "📋 复制分析结果",
                            data=result_txt,
                            file_name="fft_result.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col2:
                        st.caption("💡 点击下载分析结果\n（可在其他地方使用）")
                    
                    # 保存统一的分析上下文，供 LLM 协作页使用
                    st.session_state["analysis_payload"] = {
                        "type": "fft",
                        "y_col": y_col,
                        "sampling_rate": sampling_rate,
                        "peak_magnitude": float(max(magnitude)) if len(magnitude) else None,
                        "figure_hint": "频谱图"
                    }
                    st.session_state["plot_file_path"] = full_path
                    st.session_state["plot_file_name"] = fname
                    try:
                        with st.spinner("正在执行双对数拟合..."):
                            k, C, r_squared, k_err, C_err, chi2r = analyzer.power_fit(
                                x_data, y_data, y_err=y_err
                            )

                            st.subheader("📈 双对数（幂律）拟合结果")
                            c1, c2, c3 = st.columns(3)
                            k_str, k_unc = analyzer.format_with_uncertainty(k, k_err)
                            C_str, C_unc = analyzer.format_with_uncertainty(C, C_err)
                            c1.metric("幂指数 k", f"{k_str} ± {k_unc}")
                            c2.metric("系数 C", f"{C_str} ± {C_unc}")
                            c3.metric("加权 R²", f"{r_squared:.4f}")
                            st.caption(f"Reduced $\\chi^2$ = {chi2r:.3f}")

                            # 保存图像到本地，命名：双对数拟合+时间戳
                            plots_dir = os.path.join(os.getcwd(), "plots")
                            os.makedirs(plots_dir, exist_ok=True)
                            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                            fname = f"双对数拟合_{ts}.png"
                            full_path = os.path.join(plots_dir, fname)

                            plot_data = analyzer.plot_power_fit(
                                x_data, y_data,
                                title=f"{x_col} vs {y_col} Power-Law Fit",
                                xlabel=xlabel_in,
                                ylabel=ylabel_in,
                                x_err=x_err,
                                y_err=y_err,
                                k=k,
                                C=C,
                                r_squared=r_squared,
                                k_err=k_err,
                                C_err=C_err,
                                save_path=full_path
                            )

                            st.image(f"data:image/png;base64,{plot_data}", caption="双对数拟合图")

                            st.session_state.analysis_result = {
                                "type": "power_fit",
                                "k": k,
                                "C": C,
                                "r_squared": r_squared,
                                "k_err": k_err,
                                "C_err": C_err,
                                "plot_data": plot_data
                            }
                            st.success("✅ 分析完成！")
                            st.session_state["plot_file_path"] = full_path
                            st.session_state["plot_file_name"] = fname
                            
                            # 第一步：突出显示下一步提示
                            st.divider()
                            st.success("🎉 第二阶段完成！双对数拟合已生成！")
                            with st.container(border=True):
                                st.markdown("### 💡 下一步")
                                st.markdown("""
                                在左侧菜单选择 **'LLM协作'** 与大语言模型共同撰写实验总结
                                
                                - 📝 Plan阶段：澄清实验背景和方法
                                - ✍️ Act阶段：生成结果分析和讨论
                                """)
                            
                            # 第二步：复制按钮（作为辅助功能）
                            st.divider()
                            st.subheader("📋 复制拟合结果（可选）")
                            col1, col2 = st.columns(2)
                            with col1:
                                result_txt = f"""双对数拟合结果：
幂指数: {k_str} ± {k_unc}
系数C: {C_str} ± {C_unc}
加权R²: {r_squared:.4f}
Reduced χ²: {chi2r:.3f}
图文件: {fname}"""
                                st.download_button(
                                    "📋 复制拟合结果",
                                    data=result_txt,
                                    file_name="power_fit_result.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            with col2:
                                st.caption("💡 点击下载拟合结果\n（可在其他地方使用）")
                            
                            # 保存统一的分析上下文，供 LLM 协作页使用
                            st.session_state["analysis_payload"] = {
                                "type": "power",
                                "x_col": x_col,
                                "y_col": y_col,
                                "k": k,
                                "C": C,
                                "k_err": k_err,
                                "C_err": C_err,
                                "r_squared": r_squared,
                                "reduced_chi2": chi2r,
                                "figure_hint": "双对数拟合图"
                            }
                    except ValueError as e:
                        st.error(f"❌ 双对数拟合失败: {e}")
    else:
        st.warning("当前数据不包含足够的数值列进行分析")
        st.info("请确保表格包含至少2列数值数据")

## 已移除 LaTeX 输出页面，统一在“LLM协作”页完成协作式生成

def llm_page():
    st.header("LLM协作")
    st.caption("身份：物理实验报告协作助手｜规则：简洁凝练、禁止编造、先澄清后动笔")

    df = st.session_state.get("dataframe")
    analysis_payload = st.session_state.get("analysis_payload")
    plot_file_path = st.session_state.get("plot_file_path")
    plot_file_name = st.session_state.get("plot_file_name")

    with st.expander("📄 可用表格数据（预览）", expanded=False):
        if df is not None and not df.empty:
            st.dataframe(df)
        else:
            st.info("暂无表格数据，请先完成 OCR 并保存表格。")

    with st.expander("📊 可用分析结果（JSON）", expanded=False):
        if analysis_payload:
            st.json(analysis_payload)
        else:
            st.info("暂无分析结果，请先在‘数据分析’中完成拟合。")

    mode = st.radio("协作模式", ["plan", "act"], horizontal=True, help="plan：澄清需求与制定计划；act：按需输出文段（只包含文字分析，不用LLM生成表格/图片）")
    user_notes = st.text_area("补充说明 / 额外要求（可选）", height=100)

    llm = LLMProcessor(model=st.session_state.selected_model)

    if mode == "plan":
        if st.button("生成计划与问题清单"):
            with st.spinner("AI 正在制定计划并确认问题..."):
                resp = llm.generate_collab_response(df, analysis_payload, mode="plan", user_notes=user_notes)
                st.subheader("🧭 协作计划 & 需确认问题")
                st.write(resp)
    else:
        if st.button("执行 act 输出（文字分析）"):
            with st.spinner("AI 正在生成文字分析..."):
                resp = llm.generate_collab_response(df, analysis_payload, mode="act", act_type="text", user_notes=user_notes)
                st.subheader("📝 结果与讨论（文字分析）")
                st.write(resp)

    st.divider()
    st.subheader("📋 标准 LaTeX 表格片段（无需LLM）")
    if df is not None and not df.empty:
        from latex_generator import LatexGenerator
        generator = LatexGenerator()
        table_caption = "实验数据表"
        table_label = "tab:data"
        latex_table = generator.generate_table_latex(df, table_caption, table_label)
        st.code(latex_table, language="latex")
    else:
        st.info("暂无表格数据")

    st.subheader("�️ 标准 LaTeX 插图引用片段（无需LLM）")
    if plot_file_name and analysis_payload:
        figure_caption = analysis_payload.get("figure_hint", "实验图")
        figure_latex = f"""
\begin{{figure}}[h]
\centering
\includegraphics[width=0.8\linewidth]{{{plot_file_name}}}
\caption{{{figure_caption}}}
\label{{fig:plot}}
\end{{figure}}
"""
        st.code(figure_latex.strip(), language="latex")
        # 提供图片下载（文件名与 LaTeX 中一致）
        try:
            with open(plot_file_path, "rb") as f:
                st.download_button("📥 下载图像", data=f.read(), file_name=plot_file_name, mime="image/png")
        except Exception:
            st.warning("图像文件不可用，请重新在数据分析页生成图像。")
    else:
        st.info("暂无生成的图像，先在数据分析页执行拟合。")

def llm_page_new():
    st.header("💬 LLM 协作")
    st.caption("沟通(Plan) → 撰写(Act) → 查看结果")
    
    # 模型选择
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🤖 语言模型选择")
    with col2:
        model_options = {
            "qwen-flash-2025-07-28": "⚡ Flash (快速)",
            "qwen3-max-2025-09-23": "💪 Max (高性能)",
            "qwen-plus-2025-12-01": "🧠 Plus (深度思考)"
        }
        selected_label = st.radio(
            "选择模型",
            options=list(model_options.values()),
            index=list(model_options.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            label_visibility="collapsed"
        )
        # 反向查找选中的模型名
        st.session_state.selected_model = [k for k, v in model_options.items() if v == selected_label][0]
    
    st.divider()

    df = st.session_state.get("dataframe")
    analysis_payload = st.session_state.get("analysis_payload")
    plot_file_path = st.session_state.get("plot_file_path")
    plot_file_name = st.session_state.get("plot_file_name")
    
    # 状态
    in_act_mode = st.session_state.get("in_act_mode", False)
    act_completed = st.session_state.get("act_completed", False)
    
    # 左侧数据概览
    with st.sidebar:
        st.divider()
        st.subheader("📊 数据")
        if df is not None:
            st.caption(f"表: {len(df)} 行 × {len(df.columns)} 列")
            with st.expander("查看", expanded=False):
                st.dataframe(df, height=120, use_container_width=True)
        
        st.divider()
        st.subheader("🎯 误差分析")
        if st.session_state.get("uncertainty_analysis_result"):
            unc_result = st.session_state.uncertainty_analysis_result
            st.success("✅ 已完成计算")
            st.metric(
                "结果",
                f"{unc_result.get('result', 0):.4g} ± {unc_result.get('uncertainty_total', 0):.3g}"
            )
            st.metric(
                "相对不确定度",
                f"{unc_result.get('relative_uncertainty', 0):.2%}"
            )
            with st.expander("详细信息", expanded=False):
                st.markdown(f"**公式**: {unc_result.get('raw_formula', 'N/A')}")
                st.markdown(f"**规范式**: {unc_result.get('normalized_formula', 'N/A')}")
                if unc_result.get('contributions'):
                    st.markdown("**贡献占比**:")
                    for var, contrib in unc_result['contributions'].items():
                        st.markdown(f"  - {var}: {contrib:.1f}%")
                if unc_result.get('summary'):
                    st.markdown("---")
                    st.markdown("**AI总结**:")
                    st.markdown(unc_result['summary'])
        else:
            st.info("暂无计算结果")
        
        st.divider()
        st.subheader("📈 拟合结果")
        if analysis_payload:
            fit_type = analysis_payload.get('type', '未知')
            type_names = {
                'linear': '🔵 线性拟合',
                'log': '🟠 对数拟合',
                'fft': '🟡 傅里叶变换',
                'power': '🔴 双对数拟合'
            }
            st.success(f"✅ {type_names.get(fit_type, fit_type)}")
            
            with st.expander("参数详情", expanded=True):
                analyzer = DataAnalyzer()
                
                if fit_type == 'linear':
                    slope = analysis_payload.get('slope', 0)
                    intercept = analysis_payload.get('intercept', 0)
                    slope_err = analysis_payload.get('slope_err', 0)
                    intercept_err = analysis_payload.get('intercept_err', 0)
                    r_squared = analysis_payload.get('r_squared', 0)
                    
                    slope_str, slope_unc = analyzer.format_with_uncertainty(slope, slope_err)
                    intercept_str, intercept_unc = analyzer.format_with_uncertainty(intercept, intercept_err)
                    
                    st.metric("斜率", f"{slope_str} ± {slope_unc}")
                    st.metric("截距", f"{intercept_str} ± {intercept_unc}")
                    st.metric("加权 R²", f"{r_squared:.4f}")
                    
                elif fit_type == 'log':
                    a = analysis_payload.get('a', 0)
                    b = analysis_payload.get('b', 0)
                    a_err = analysis_payload.get('a_err', 0)
                    b_err = analysis_payload.get('b_err', 0)
                    r_squared = analysis_payload.get('r_squared', 0)
                    
                    a_str, a_unc = analyzer.format_with_uncertainty(a, a_err)
                    b_str, b_unc = analyzer.format_with_uncertainty(b, b_err)
                    
                    st.metric("系数 a", f"{a_str} ± {a_unc}")
                    st.metric("截距 b", f"{b_str} ± {b_unc}")
                    st.metric("加权 R²", f"{r_squared:.4f}")
                    
                elif fit_type == 'fft':
                    sampling_rate = analysis_payload.get('sampling_rate', 'N/A')
                    peak_magnitude = analysis_payload.get('peak_magnitude', 'N/A')
                    st.metric("采样率", f"{sampling_rate} Hz")
                    if isinstance(peak_magnitude, (int, float)):
                        st.metric("峰值幅度", f"{peak_magnitude:.6g}")
                    else:
                        st.metric("峰值幅度", f"{peak_magnitude}")
                        
                elif fit_type == 'power':
                    k = analysis_payload.get('k', 0)
                    C = analysis_payload.get('C', 0)
                    k_err = analysis_payload.get('k_err', 0)
                    C_err = analysis_payload.get('C_err', 0)
                    r_squared = analysis_payload.get('r_squared', 0)
                    
                    k_str, k_unc = analyzer.format_with_uncertainty(k, k_err)
                    C_str, C_unc = analyzer.format_with_uncertainty(C, C_err)
                    
                    st.metric("幂指数 k", f"{k_str} ± {k_unc}")
                    st.metric("系数 C", f"{C_str} ± {C_unc}")
                    st.metric("加权 R²", f"{r_squared:.4f}")
                    st.metric("R²", f"{analysis_payload.get('r_squared', 'N/A'):.4f}")
            
            if plot_file_name:
                st.info(f"🖼️ 图: {plot_file_name}")
        else:
            st.warning("⚠️ 请先完成数据分析")
        
        st.divider()
        if st.button("🔄 重置", use_container_width=True):
            for k in ["chat_history", "context_initialized", "in_act_mode", "act_completed", "act_response"]:
                st.session_state.pop(k, None)
            st.rerun()
    
    # LLM 初始化
    try:
        llm = LLMProcessor(model=st.session_state.selected_model)
    except ValueError as e:
        st.error(f"❌ {str(e)}")
        return
    
    # ===== PLAN 阶段 =====
    if not in_act_mode and not act_completed:
        # 初始化
        if not st.session_state.get("context_initialized", False):
            init = "你好！我是物理实验报告助手。已读取你的数据和分析。请告诉我：\n1. 实验名称和目标\n2. 测量方法\n3. 关键参数\n4. 验证的物理规律"
            st.session_state.chat_history = [{"role": "assistant", "content": init}]
            st.session_state.context_initialized = True
        
        # 显示对话
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])
        
        # 输入
        if inp := st.chat_input("描述实验..."):
            st.chat_message("user", avatar="👤").markdown(inp)
            st.session_state.chat_history.append({"role": "user", "content": inp})
            
            ctx = {"dataframe": df, "analysis_payload": analysis_payload} if len(st.session_state.chat_history) <= 3 else None
            
            with st.chat_message("assistant", avatar="🤖"):
                ph = st.empty()
                resp = ""
                thinking_text = ""
                # 判断是否使用plus模型并启用深度思考
                enable_thinking = "plus" in st.session_state.selected_model
                try:
                    for chunk_obj in llm.chat_stream(inp, st.session_state.chat_history[:-1], ctx, enable_thinking=enable_thinking):
                        if isinstance(chunk_obj, dict):
                            chunk_type = chunk_obj.get("type", "")
                            chunk_text = chunk_obj.get("text", "")
                            
                            if chunk_type == "thinking":
                                thinking_text += chunk_text
                            elif chunk_type == "content":
                                resp += chunk_text
                        else:
                            # 向后兼容：如果返回字符串而不是字典
                            resp += str(chunk_obj)
                        
                        # 实时显示内容和思考过程
                        if thinking_text and resp:
                            display_text = f"🧠 **思考过程**\n\n```\n{thinking_text}\n```\n\n---\n\n{resp}▌"
                        elif thinking_text:
                            display_text = f"🧠 **思考过程**\n\n```\n{thinking_text}\n```▌"
                        else:
                            display_text = resp + "▌"
                        ph.markdown(display_text)
                    
                    # 最终显示（去除光标）
                    if thinking_text and resp:
                        display_text = f"🧠 **思考过程**\n\n```\n{thinking_text}\n```\n\n---\n\n{resp}"
                    elif thinking_text:
                        display_text = f"🧠 **思考过程**\n\n```\n{thinking_text}\n```"
                    else:
                        display_text = resp
                    ph.markdown(display_text)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
                except Exception as e:
                    ph.error(f"❌ {e}")
        
        # Act 按钮
        st.divider()
        col1, col2 = st.columns([2, 3])
        with col1:
            if st.button("✅ 开始撰写", use_container_width=True, type="primary"):
                st.session_state.in_act_mode = True
                st.rerun()
        with col2:
            st.caption("💡 沟通完成后，点击进入撰写阶段")
    
    # ===== ACT 阶段 =====
    elif in_act_mode and not act_completed:
        st.subheader("📝 撰写报告片段")
        st.caption("根据沟通内容生成精简的实验报告片段（非完整报告）")
        
        mod = st.text_area("修改要求（可选）", placeholder="例：加上不确定度讨论", height=60)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🚀 生成", use_container_width=True):
                try:
                    with st.chat_message("assistant", avatar="🤖"):
                        ph = st.empty()
                        resp = ""
                        thinking_text = ""
                        # 判断是否使用plus模型并启用深度思考
                        enable_thinking = "plus" in st.session_state.selected_model
                        uncertainty_result = st.session_state.get("passed_uncertainty_result")
                        uncertainty_conversation = st.session_state.get("passed_uncertainty_conversation")
                        
                        for chunk_obj in llm.generate_act_response(df, analysis_payload, st.session_state.get("chat_history", []), mod, enable_thinking=enable_thinking, uncertainty_result=uncertainty_result, uncertainty_conversation=uncertainty_conversation):
                            if isinstance(chunk_obj, dict):
                                chunk_type = chunk_obj.get("type", "")
                                chunk_text = chunk_obj.get("text", "")
                                
                                if chunk_type == "thinking":
                                    thinking_text += chunk_text
                                elif chunk_type == "content":
                                    resp += chunk_text
                            else:
                                # 向后兼容：如果返回字符串而不是字典
                                resp += str(chunk_obj)
                        
                        # 最终显示（去除光标）
                        if thinking_text and resp:
                            display_text = f"🧠 **深度思考过程**\n\n```\n{thinking_text}\n```\n\n---\n\n{resp}"
                        elif thinking_text:
                            display_text = f"🧠 **深度思考过程**\n\n```\n{thinking_text}\n```"
                        else:
                            display_text = resp
                        ph.markdown(display_text)
                        
                        st.session_state.act_response = resp
                        st.session_state.act_completed = True
                        st.success("🎉 报告片段已生成！")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
        with col2:
            if st.button("↩️ 回到沟通", use_container_width=True):
                st.session_state.in_act_mode = False
                st.rerun()
    
    # ===== 结果展示 =====
    elif act_completed:
        st.subheader("✨ 结果")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📝 报告", "📋 表", "🖼️ 图", "📥 下载"])
        
        with tab1:
            st.caption("生成的报告片段")
            st.code(st.session_state.get("act_response", ""), language="latex")
            if st.button("✏️ 重新生成"):
                st.session_state.in_act_mode = True
                st.session_state.act_completed = False
                st.rerun()
        
        with tab2:
            if df is not None:
                from latex_generator import LatexGenerator
                gen = LatexGenerator()
                tbl = gen.generate_table_latex(df, "实验数据表", "tab:data")
                st.code(tbl, language="latex")
            else:
                st.info("暂无数据")
        
        with tab3:
            if plot_file_name and analysis_payload:
                fig_cap = analysis_payload.get("figure_hint", "拟合")
                fig_tex = f"""\\\\begin{{figure}}[h]
\\\\centering
\\\\includegraphics[width=0.8\\\\linewidth]{{{plot_file_name}}}
\\\\caption{{{fig_cap}}}
\\\\label{{fig:plot}}
\\\\end{{figure}}"""
                st.code(fig_tex, language="latex")
            else:
                st.info("暂无图表")
        
        with tab4:
            if plot_file_path and plot_file_name:
                try:
                    with open(plot_file_path, "rb") as f:
                        st.download_button(f"📥 {plot_file_name}", data=f.read(), file_name=plot_file_name, mime="image/png", use_container_width=True)
                except:
                    st.warning("图像不可用")
        
        st.divider()
        st.subheader("📦 完整报告包下载")
        st.caption("包含：生成的报告片段（LaTeX） + 数据表（LaTeX） + 图像（PNG）+ 数据（CSV）")
        
        if st.button("⬇️ 生成并下载完整包", use_container_width=True):
            try:
                import zipfile
                import io
                
                # 创建 ZIP 文件（在内存中）
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # 1. 添加生成的报告片段
                    act_resp = st.session_state.get("act_response", "")
                    if act_resp:
                        zf.writestr("report_segment.tex", act_resp)
                    
                    # 2. 添加数据表的 LaTeX
                    if df is not None:
                        from latex_generator import LatexGenerator
                        gen = LatexGenerator()
                        table_latex = gen.generate_table_latex(df, "实验数据表", "tab:data")
                        zf.writestr("data_table.tex", table_latex)
                        
                        # 3. 添加数据（CSV）
                        csv_data = df.to_csv(index=False)
                        zf.writestr("data_table.csv", csv_data)
                    
                    # 4. 添加图像
                    if plot_file_path:
                        try:
                            with open(plot_file_path, "rb") as f:
                                zf.writestr(plot_file_name or "plot.png", f.read())
                        except:
                            pass
                    
                    # 5. 添加自定义 LaTeX（如有）
                    custom_latex = st.session_state.get("custom_latex", "")
                    if custom_latex:
                        zf.writestr("custom_content.tex", custom_latex)
                    
                    # 6. 添加 README
                    readme = """# 物理实验报告文件包

此包包含以下文件：

1. **report_segment.tex** - AI 生成的报告片段
2. **data_table.tex** - 实验数据表（LaTeX 格式）
3. **data_table.csv** - 实验数据表（CSV 格式）
4. **plot.png** - 拟合曲线或分析图像
5. **custom_content.tex** - 自定义 LaTeX 内容（如有）
6. **README.md** - 本文件

## 使用说明

1. 将这些文件复制到您的 LaTeX 项目目录
2. 在主文档中使用 \\input{report_segment.tex} 导入报告片段
3. 根据需要调整图像大小和位置
4. 编译生成 PDF

## 注意

- 确保 \\includegraphics 命令中的文件名与实际文件匹配
- 可能需要调整表格和图像的格式以符合您的文档风格
- 自定义 LaTeX 内容可能需要进一步编辑
"""
                    zf.writestr("README.md", readme)
                
                # 准备下载
                zip_buffer.seek(0)
                st.download_button(
                    "📦 下载完整报告包（ZIP）",
                    data=zip_buffer.getvalue(),
                    file_name="physics_report_package.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )
                st.success("🎉 **第三阶段完成！** 完整报告包已准备好！包含：报告片段 + 数据表 + 图像 + CSV数据")
                st.balloons()
                st.success("✅ 完整包已准备好！包含报告、表格、图像和数据。")
            except Exception as e:
                st.error(f"❌ 打包失败: {e}")
        
        st.divider()
        if st.button("🔄 新实验", use_container_width=True):
            for k in ["chat_history", "context_initialized", "in_act_mode", "act_completed", "act_response"]:
                st.session_state.pop(k, None)
            st.rerun()

if __name__ == "__main__":
    main()
