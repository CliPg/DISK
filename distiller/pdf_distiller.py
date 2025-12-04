import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import os
from paddleocr import PaddleOCR
from PIL import Image
import io
import numpy as np
import pdf2image
import re

class PDFDistiller:
    
    def __init__(self):
        # self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
        pass

    def extract_text_blocks(self, pdf_path:str)->list:
        """
        extract blocks per page from pdf.
        one block usually corresponds to one paragraph.

        Args:
            pdf_path (str): path to the pdf file

        Returns:
            list: list of paragraphs extracted from the pdf
        """
        doc = fitz.open(pdf_path)
        all_paragraphs = []

        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                text = str(b[4])
                if self.is_valid_block(text.strip()):
                    all_paragraphs.append(text.strip())
                

        """
        Example: Save extracted paragraphs to a text file for verification
        """
        with open("../logs/extracted_paragraphs.log", "w", encoding="utf-8") as f:
            for para in all_paragraphs:
                f.write(para + "\n\n")

        return all_paragraphs
    
    def is_valid_block(self, text: str) -> bool:
        """
        Determine if a text block is valid based on certain criteria.
        
        Args:
            text (str): The text block to evaluate.

        Returns:
            bool: True if the block is valid, False otherwise.
        """

        if not text:
            return False

        # minimum criteria for a valid text block
        if len(text) < 10:
            return False

        # must contain at least one alphabetic character (English or Chinese)
        if not re.search(r"[A-Za-z\u4e00-\u9fa5]", text):
            return False

        # must contain more than two words
        if len(text.split()) <= 2:
            return False
        
        # exclude reference format
        if self._is_reference_format(text):
            return False
        
        # must be a complete sentence
        if self._is_a_sentence(text) == False:
            return False
        
        return True
    
    def _is_reference_format(self, text: str) -> bool:
        """
        Judge whether a given text block is likely to be a reference entry.
        
        Args:
            text (str): The text block to evaluate.

        Returns:
            bool: True if the block is likely a reference, False otherwise.
        """

        # e.g.[10] Huasheng Liu, ...
        pattern1 = r"\s*\[\s*\d+\s*\]\s*[A-Z][a-z]+(\s+([A-Z]\.?|\w+))?"

        # e.g. Liyi Chen1,2†, Panrong Tong2, Zhongming Jin2, Ying Sun3, Jieping Ye2∗, Hui Xiong3,4∗
        pattern2 = r"^(\s*[A-Z][A-Za-z\s]+[\d\s,\*\†\-]*\s*){3,}"
                
        # some key words（like 'Proceedings', 'Journal', 'Conference', 'pages', 'vol'）
        reference_keywords = r"(Proceedings|Journal|Conference|pages|vol|ACM|IEEE|arXiv|Springer)"
        
        if re.search(pattern1, text):
            with open("../logs/distill_references.log", "a", encoding="utf-8") as f:
                f.write(text + "\n\n")
            return True
        
        if re.search(reference_keywords, text, re.IGNORECASE) and re.search(r"[A-Z]\.?,\s*[A-Z]", text):
             return True

        return False
    
    def _is_a_sentence(self, text: str) -> bool:
        """
        Judge whether a given text block is likely to be a complete sentence.
        
        Args:
            text (str): The text block to evaluate.

        Returns:
            bool: True if the block is likely a complete sentence, False otherwise.
        """
        # Check if the text ends with a sentence-ending punctuation
        if re.search(r"[\.\?\!\。]", text.strip()):
            return True
        
        with open("../logs/distill_incomplete_sentences.log", "a", encoding="utf-8") as f:
            f.write(text + "\n\n")

        return False

    # TODO: need to be improved
    def extract_images_and_ocr(self, pdf_path):
        doc = fitz.open(pdf_path)
        results = []

        for page_index, page in enumerate(doc):
            # 使用 page.get_images() 提取图片对象
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # --- 优化点 1: 确保图片格式正确加载 ---
                try:
                    # 使用 Image.open 加载图片
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    img_np = np.array(pil_img)
                except Exception as e:
                    print(f"Page {page_index+1}, Image {img_index}: Failed to open image bytes: {e}")
                    continue # 跳过无法打开的图片

                # --- 优化点 2: 修正 PaddleOCR 调用和结果解析 ---
                # 假设 ocr_model 是 PaddleOCR 实例，调用 ocr 方法
                # 注意：PaddleOCR.ocr() 接收 np.array 列表或路径列表。
                try:
                    # PaddleOCR 默认返回格式：[[[bbox], [text, confidence]], ...]
                    # 推荐使用 ocr_model.ocr() 方法
                    ocr_results = self.ocr_model.predict(img_np) 

                    # 检查结果是否为空或格式不正确
                    if ocr_results and ocr_results[0] is not None:
                        # 结果在 ocr_results[0] 中 (因为只处理了一张图片)
                        ocr_lines = ocr_results[0]

                        # 安全提取文本： line[1] 是 [text, confidence] 列表
                        text = "\n".join([line[1][0] for line in ocr_lines if line and len(line) > 1 and len(line[1]) > 0])
                    else:
                        text = ""

                except Exception as e:
                    # 捕获 OCR 过程中的异常
                    print(f"Page {page_index+1}, Image {img_index}: OCR failed with error: {e}")
                    text = "OCR Error"

                results.append({
                    "page": page_index + 1,
                    "image": pil_img,
                    "ocr_text": text
                })

        return results
    
    # TODO: need to be improved
    def extract_tables(self, pdf_path):
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for tb in page_tables:
                    df = pd.DataFrame(tb)
                    tables.append(df)


        """
        Example: Save extracted tables to CSV for verification
        """
        # with open("extracted_tables.csv", "w") as f:
        #     for i, table in enumerate(tables):
        #         f.write(f"Table {i+1}:\n")
        #         table.to_csv(f, index=False)
        #         f.write("\n")

        return tables
        
    # TODO: need to be improved
    def extract_tables_improved(self, pdf_path):
        tables = []

        # 调整默认设置
        table_settings = {
            # 1. 调整识别水平和垂直线的阈值（核心）
            "vertical_strategy": "lines",  # 默认值，也可以尝试 "text"
            "horizontal_strategy": "lines", # 默认值，也可以尝试 "text"
            
            # 2. 调整线合并和识别的公差（针对模糊或不完整的线条）
            "snap_tolerance": 3,  # 增加/减少合并相近线的公差
            "join_tolerance": 3,  # 增加/减少连接断线的公差

            # 3. 如果表格没有线条，尝试基于文本的边界识别
            # "edge_min_length": 5, # 只有在没有线条时才启用
        }

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 使用调整后的设置
                page_tables = page.extract_tables(table_settings=table_settings)
                
                for tb in page_tables:
                    # 确保提取的内容不为空
                    if tb and any(row for row in tb if row):
                        df = pd.DataFrame(tb)
                        tables.append(df)
        
        with open("extracted_tables_improved.csv", "w") as f:
            for i, table in enumerate(tables):
                f.write(f"Table {i+1}:\n")
                table.to_csv(f, index=False)
                f.write("\n")

        return tables
    
    # TODO: need to be improved
    def extract_tables_horizontal_lines(self, pdf_path):
        tables = []

        # 核心设置：依赖横线 ('lines') 识别行，依赖文本间距 ('text') 识别列
        table_settings = {
            "vertical_strategy": "text",  # 基于文本对齐和间距来确定列边界
            "horizontal_strategy": "lines", # 基于实际的横线来确定行边界
            
            # 调整公差，帮助识别不完全对齐的横线
            "snap_tolerance": 5,  
            "join_tolerance": 5, 
            
            # 确保文本策略的列识别能够适应您的文本对齐
            # "text_y_tolerance": 3, 
        }

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables(table_settings=table_settings)
                
                for tb in page_tables:
                    if tb and any(row for row in tb if row):
                        df = pd.DataFrame(tb)
                        tables.append(df)

        with open("extracted_tables_horizontal_lines.csv", "w") as f:
            for i, table in enumerate(tables):
                f.write(f"Table {i+1}:\n")
                table.to_csv(f, index=False)
                f.write("\n")
        
        return tables
            