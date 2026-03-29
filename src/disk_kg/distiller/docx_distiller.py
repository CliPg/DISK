import io
import re
import zipfile

import docx2txt
import jieba
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .distiller import Distiller


class DocxDistiller(Distiller):
    """
    Distiller for .docx files.
    Uses docx2txt for text/image extraction (no python-docx dependency).
    Supports OCR on embedded images via RapidOCR.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.ocr_model = RapidOCR(use_angle_cls=True, lang="en")

    def extract_text_blocks(self) -> list[str]:
        """
        Extract text blocks (paragraphs) from a .docx file using docx2txt.
        Blocks shorter than 10 characters are merged with the next block.

        Returns:
            list: List of paragraphs extracted from the document.
        """
        full_text = docx2txt.process(self.file_path)
        raw_blocks = []

        for line in full_text.split("\n"):
            text = line.strip()
            if text:
                raw_blocks.append(text)

        # 合并短块：长度小于10的合并到下一个块
        merged_paragraphs = []
        i = 0
        while i < len(raw_blocks):
            current = raw_blocks[i]
            if len(current) < 10 and i + 1 < len(raw_blocks):
                current += " " + raw_blocks[i + 1]
                i += 1
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
        """
        pattern1 = r"\s*\[\s*\d+\s*\]\s*[A-Z][a-z]+(\s+([A-Z]\.?|\w+))?"
        pattern2 = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
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
        """
        if re.search(r"[\.\?\!\。]", text.strip()):
            return True

        return False

    def extract_images_and_ocr(self) -> list[str]:
        """
        Extract images embedded in the .docx file and perform OCR on them.

        Returns:
            list[str]: List of OCR text strings from images.
        """
        results = []
        img_index = 0

        with zipfile.ZipFile(self.file_path, "r") as z:
            image_entries = [
                name
                for name in z.namelist()
                if name.startswith("word/media/")
                and any(
                    name.lower().endswith(ext)
                    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif")
                )
            ]

            for entry in sorted(image_entries):
                try:
                    image_bytes = z.read(entry)
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    img_np = np.array(pil_img)
                except Exception as e:
                    print(f"Image {img_index} ({entry}): Failed to open image: {e}")
                    img_index += 1
                    continue

                # OCR
                try:
                    ocr_result, _ = self.ocr_model(img_np)
                    if ocr_result:
                        text = "\n".join([line[1] for line in ocr_result if line])
                    else:
                        text = ""
                except Exception as e:
                    print(f"Image {img_index} ({entry}): OCR failed with error: {e}")
                    text = "OCR Error"

                results.append(
                    {
                        "index": img_index,
                        "image": pil_img,
                        "ocr_text": text,
                    }
                )
                img_index += 1

        return results

    def extract_tables(self) -> list[str]:
        """
        Extract tables from a .docx file and return as list of Markdown strings.
        """
        import xml.etree.ElementTree as ET

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        tables = []

        with zipfile.ZipFile(self.file_path, "r") as z:
            if "word/document.xml" not in z.namelist():
                return tables
            xml_content = z.read("word/document.xml")

        root = ET.fromstring(xml_content)

        for tbl in root.iter(f"{{{ns['w']}}}tbl"):
            table_data = []
            for tr in tbl.iter(f"{{{ns['w']}}}tr"):
                row_data = []
                for tc in tr.iter(f"{{{ns['w']}}}tc"):
                    # Collect all text within the cell
                    cell_texts = []
                    for t in tc.iter(f"{{{ns['w']}}}t"):
                        if t.text:
                            cell_texts.append(t.text)
                    row_data.append("".join(cell_texts).strip())
                table_data.append(row_data)
            if table_data:
                tables.append(table_data)

        return tables
