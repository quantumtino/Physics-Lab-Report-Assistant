import pandas as pd
from jinja2 import Template
import numpy as np

class LatexGenerator:
    def __init__(self):
        # LaTeX 表格模板
        self.table_template = Template('''
\\begin{table}[h]
\\centering
\\begin{tabular}{|{{ column_alignments }}|}
\\hline
{% for header in headers %}
{{ header }}{% if not loop.last %} & {% endif %}
{% endfor %} \\\\
\\hline
{% for row in rows %}
{% for cell in row %}
{{ cell }}{% if not loop.last %} & {% endif %}
{% endfor %} \\\\
{% endfor %}
\\hline
\\end{tabular}
\\caption{ {{ caption }} }
\\label{tab:{{ label }}}
\\end{table}
        ''')
        
        # LaTeX 图表模板
        self.figure_template = Template('''
\\begin{figure}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{ {{ image_path }} }
\\caption{ {{ caption }} }
\\label{fig:{{ label }}}
\\end{figure}
        ''')
        
        # LaTeX 报告主模板
        self.report_template = Template('''
\\documentclass[12pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{array}
\\usepackage{geometry}
\\geometry{a4paper, margin=1in}

\\title{ {{ title }} }
\\author{ {{ author }} }
\\date{ {{ date }} }

\\begin{document}

\\maketitle

\\begin{abstract}
{{ abstract }}
\\end{abstract}

\\section{实验目的}
{{ purpose }}

\\section{实验原理}
{{ theory }}

\\section{实验数据与处理}
{{ data_section }}

\\section{结果分析}
{{ analysis }}

\\section{结论}
{{ conclusion }}

\\end{document}
        ''')

    def generate_table_latex(self, df: pd.DataFrame, caption: str = "Table", label: str = "tab:mylabel") -> str:
        """
        将DataFrame转换为LaTeX表格
        :param df: 包含数据的DataFrame
        :param caption: 表格标题
        :param label: 表格标签
        :return: LaTeX格式的表格字符串
        """
        # 获取列数并生成对齐字符串
        num_columns = len(df.columns)
        column_alignments = 'c|' * num_columns  # 使用居中对齐
        
        # 准备表格数据
        headers = [str(col) for col in df.columns]
        rows = []
        for index, row in df.iterrows():
            rows.append([str(cell) for cell in row])
        
        # 渲染模板
        latex_table = self.table_template.render(
            column_alignments=column_alignments,
            headers=headers,
            rows=rows,
            caption=caption,
            label=label
        )
        
        return latex_table
    
    def generate_figure_latex(self, image_path: str, caption: str = "Figure", label: str = "fig:mylabel") -> str:
        """
        生成LaTeX图表代码
        :param image_path: 图像路径
        :param caption: 图表标题
        :param label: 图表标签
        :return: LaTeX格式的图表字符串
        """
        latex_figure = self.figure_template.render(
            image_path=image_path,
            caption=caption,
            label=label
        )
        
        return latex_figure
    
    def generate_report_latex(self, title: str, author: str, date: str, 
                             abstract: str, purpose: str, theory: str,
                             data_section: str, analysis: str, conclusion: str) -> str:
        """
        生成完整LaTeX报告
        :param title: 报告标题
        :param author: 作者
        :param date: 日期
        :param abstract: 摘要
        :param purpose: 实验目的
        :param theory: 实验原理
        :param data_section: 数据部分
        :param analysis: 结果分析
        :param conclusion: 结论
        :return: 完整的LaTeX报告字符串
        """
        latex_report = self.report_template.render(
            title=title,
            author=author,
            date=date,
            abstract=abstract,
            purpose=purpose,
            theory=theory,
            data_section=data_section,
            analysis=analysis,
            conclusion=conclusion
        )
        
        return latex_report