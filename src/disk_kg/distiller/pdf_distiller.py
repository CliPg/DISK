import io
import re

import fitz  # PyMuPDF
import jieba
import numpy as np
import pandas as pd
import pdfplumber
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .distiller import Distiller


class PDFDistiller(Distiller):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.ocr_model = RapidOCR(use_angle_cls=True, lang="en")

    def extract_text_blocks(self) -> list[str]:
        """
        extract blocks per page from pdf.
        one block usually corresponds to one paragraph.
        blocks shorter than 10 characters are merged with the next block.

        Returns:
            list: list of paragraphs extracted from the pdf
        """
        doc = fitz.open(self.file_path)
        raw_blocks = []

        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                text = str(b[4]).strip()
                if text:  # 收集所有非空文本块
                    raw_blocks.append(text)

        # 合并短块：长度小于10的合并到下一个块
        merged_paragraphs = []
        i = 0
        while i < len(raw_blocks):
            current = raw_blocks[i]
            # 如果当前块长度小于10，且不是最后一个块，则合并到下一个块
            if len(current) < 10 and i + 1 < len(raw_blocks):
                current += " " + raw_blocks[i + 1]
                i += 1  # 跳过下一个块，因为已经合并了
            if self.is_valid_block(current):
                merged_paragraphs.append(current)
            i += 1

        return merged_paragraphs

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

        # must contain at least one alphabetic character (English or Chinese)
        if not re.search(r"[A-Za-z\u4e00-\u9fa5]", text):
            return False

        # must contain more than two words
        if len(text.split()) <= 2 and len(list(jieba.cut(text))) <= 2:
            return False

        # exclude reference format
        if self._is_reference_format(text):
            return False

        # must be a complete sentence
        if not self._is_a_sentence(text):
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

        return False

    def extract_images_and_ocr(self) -> list[str]:
        doc = fitz.open(self.file_path)
        results = []

        for page_index, page in enumerate(doc):  # type: ignore
            # 使用 page.get_images() 提取图片对象
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                try:
                    # 使用 Image.open 加载图片
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    img_np = np.array(pil_img)
                except Exception as e:
                    print(
                        f"Page {page_index + 1}, Image {img_index}: Failed to open image bytes: {e}"
                    )
                    continue  # 跳过无法打开的图片

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

    def extract_tables(self) -> list[str]:
        tables = []

        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for tb in page_tables:
                    df = pd.DataFrame(tb)
                    tables.append(df)

        return tables
