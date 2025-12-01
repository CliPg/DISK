import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import os
from paddleocr import PaddleOCR
from PIL import Image
import io

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

        return all_paragraphs

    def extract_images_and_ocr(self, pdf_path):
        doc = fitz.open(pdf_path)
        results = []

        for page_index, page in enumerate(doc):
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                pil_img = Image.open(io.BytesIO(image_bytes))

                # OCR 识别
                ocr_result = self.ocr_model.ocr(image_bytes, cls=True)
                text = ""
                if ocr_result:
                    text = "\n".join([line[1][0] for line in ocr_result])

                results.append({
                    "page": page_index + 1,
                    "image": pil_img,
                    "ocr_text": text
                })

        return results
