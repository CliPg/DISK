import io
import re
from pathlib import Path

import fitz  # PyMuPDF
import jieba
import numpy as np
import pandas as pd
import pdfplumber
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


class PDFDistiller:
    def __init__(self):
        self.ocr_model = RapidOCR(use_angle_cls=True, lang="en")
        # Set log directory to project root/logs
        self.log_dir = Path(__file__).resolve().parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_blocks(self, pdf_path: str) -> list:
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
        (self.log_dir / "extracted_paragraphs.log").write_text(
            "\n\n".join(all_paragraphs), encoding="utf-8"
        )

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
            print("Empty block found, skipping.")
            return False

        # minimum criteria for a valid text block
        if len(text) < 10:
            print(f"Block too short (length {len(text)}), skipping: {text}")
            return False

        # must contain at least one alphabetic character (English or Chinese)
        if not re.search(r"[A-Za-z\u4e00-\u9fa5]", text):
            print(f"Block does not contain alphabetic characters, skipping: {text}")
            return False

        # must contain more than two words
        if len(text.split()) <= 2 and len(list(jieba.cut(text))) <= 2:
            print(f"Block does not contain more than two words, skipping: {text}")
            return False

        # exclude reference format
        if self._is_reference_format(text):
            print(f"Block identified as reference format, skipping: {text}")
            return False

        # must be a complete sentence
        if not self._is_a_sentence(text):
            print(f"Block does not appear to be a complete sentence, skipping: {text}")
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

        # email
        pattern2 = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

        # some key words（like 'Proceedings', 'Journal', 'Conference', 'pages', 'vol'）
        reference_keywords = r"(Proceedings|Journal|Conference|pages|vol|ACM|IEEE|arXiv|Springer)"

        if re.search(pattern1, text) or re.search(pattern2, text):
            with (self.log_dir / "distill_references.log").open("a", encoding="utf-8") as f:
                f.write(text + "\n\n")
            return True

        if re.search(reference_keywords, text, re.IGNORECASE) and re.search(
            r"[A-Z]\.?,\s*[A-Z]", text
        ):
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

        with (self.log_dir / "distill_incomplete_sentences.log").open("a", encoding="utf-8") as f:
            f.write(text + "\n\n")

        return False

    # fixed by WuJunkai on 2026-02-23: optimize image extraction and OCR process
    def extract_images_and_ocr(self, pdf_path: str) -> list[dict]:
        doc = fitz.open(pdf_path)
        results = []

        for page_index, page in enumerate(doc):  # type: ignore
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
                    print(
                        f"Page {page_index + 1}, Image {img_index}: Failed to open image bytes: {e}"
                    )
                    continue  # 跳过无法打开的图片

                # --- 优化点 2: 修正 RapidOCR 调用和结果解析 ---
                try:
                    # RapidOCR 调用: 返回 (result, elapse)
                    ocr_result, _ = self.ocr_model(img_np)

                    if ocr_result:
                        # RapidOCR 返回格式: [[box, text, score], ...]
                        text = "\n".join([line[1] for line in ocr_result if line])
                    else:
                        text = ""

                except Exception as e:
                    # 捕获 OCR 过程中的异常
                    print(f"Page {page_index + 1}, Image {img_index}: OCR failed with error: {e}")
                    text = "OCR Error"

                results.append({"page": page_index + 1, "image": pil_img, "ocr_text": text})

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
            "horizontal_strategy": "lines",  # 默认值，也可以尝试 "text"
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
                f.write(f"Table {i + 1}:\n")
                table.to_csv(f, index=False)
                f.write("\n")

        return tables

    # TODO: need to be improved
    def extract_tables_horizontal_lines(self, pdf_path):
        tables = []

        # 核心设置：依赖横线 ('lines') 识别行，依赖文本间距 ('text') 识别列
        table_settings = {
            "vertical_strategy": "text",  # 基于文本对齐和间距来确定列边界
            "horizontal_strategy": "lines",  # 基于实际的横线来确定行边界
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
                f.write(f"Table {i + 1}:\n")
                table.to_csv(f, index=False)
                f.write("\n")

        return tables
