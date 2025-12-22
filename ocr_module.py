from paddleocr import PaddleOCR
import cv2
import numpy as np
import pandas as pd
from io import StringIO

class OCRProcessor:
    def __init__(self):
        # 初始化PaddleOCR，支持中文识别
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    
    def extract_text_from_image(self, image_path):
        """
        从图像中提取文本
        :param image_path: 图像路径
        :return: 识别结果列表
        """
        result = self.ocr.ocr(image_path)
        return result
    
    def extract_tables_from_image(self, image_path):
        """
        从图像中提取表格数据
        :param image_path: 图像路径
        :return: 表格数据的DataFrame列表
        """
        # 使用PaddleOCR的表格识别功能
        result = self.ocr.ocr(image_path)
        
        # 解析表格数据
        tables = []
        for res in result:
            if res is not None:
                for line in res:
                    if line[1][0] == 'table':  # 检测到表格
                        table_data = self.parse_table_data(line[1][1])
                        tables.append(table_data)
        
        return tables
    
    def parse_table_data(self, table_str):
        """
        解析表格字符串为DataFrame
        :param table_str: 表格字符串
        :return: DataFrame
        """
        # 将表格字符串转换为DataFrame
        try:
            df = pd.read_csv(StringIO(table_str), sep='\t')
            return df
        except Exception as e:
            print(f"解析表格数据时出错: {e}")
            return pd.DataFrame()
    
    def detect_handwritten_text(self, image_path):
        """
        检测图像中的手写文本
        :param image_path: 图像路径
        :return: 手写文本列表
        """
        # 读取图像
        img = cv2.imread(image_path)
        
        # 使用OCR识别文本
        result = self.extract_text_from_image(image_path)
        
        # 提取文本内容
        texts = []
        for res in result:
            if res is not None:
                for line in res:
                    texts.append(line[1][0])  # 提取文本内容
        
        return texts
