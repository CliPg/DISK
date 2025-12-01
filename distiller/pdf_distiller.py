import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import os
from paddleocr import PaddleOCR
from PIL import Image
import io
import numpy as np
import pdf2image

class PDFDistiller:
    
    def __init__(self):
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

    def extract_text_blocks(self, pdf_path):
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
                all_paragraphs.append(text.strip())

        """
        Example: Save extracted paragraphs to a text file for verification
        """
        with open("extracted_paragraphs.txt", "w", encoding="utf-8") as f:
            for para in all_paragraphs:
                f.write(para + "\n\n")

        return all_paragraphs

    # TODO: need to be improved
    def extract_images_and_ocr(self, pdf_path):
        doc = fitz.open(pdf_path)
        results = []

        for page_index, page in enumerate(doc):
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img_np = np.array(pil_img)

                ocr_result = self.ocr_model.predict(img_np)

                # 安全提取文本
                text = "\n".join([line[1][0] for line in ocr_result if len(line) == 2 and isinstance(line[1], (tuple, list))])

                results.append({
                    "page": page_index + 1,
                    "image": pil_img,
                    "ocr_text": text
                })

        return results
    
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
            