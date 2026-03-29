import re

import jieba

from .distiller import Distiller

try:
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import TableItem, TextItem

    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

if HAS_DOCLING:

    class DoclingDistiller(Distiller):
        """
        Distiller implementation using Docling for high-fidelity PDF and document parsing.
        Supports complex layout analysis, table extraction, and integrated OCR.
        """

        def __init__(self, file_path: str):
            super().__init__(file_path)
            # Configure pipeline options for PDF
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned content
            pipeline_options.do_table_structure = (
                True  # Enable advanced table structure recognition
            )

            # Initialize the converter with specific options
            self.converter = DocumentConverter()

        def extract_text_blocks(self) -> list[str]:
            """
            Extract text blocks (paragraphs, headers) from a document using Docling.

            Returns:
                list[str]: List of valid text paragraphs.
            """
            result = self.converter.convert(self.file_path)
            raw_blocks = []

            # Docling identifies the semantic structure (Heading, Paragraph, etc.)
            for item, _ in result.document.iterate_items():
                if isinstance(item, TextItem):
                    text = item.text.strip()
                    if text:
                        raw_blocks.append(text)

            # Merge short blocks and validate, consistent with PDFDistiller logic
            merged_paragraphs = []
            i = 0
            while i < len(raw_blocks):
                current = raw_blocks[i]
                # Merge blocks shorter than 10 characters with the next one
                if len(current) < 10 and i + 1 < len(raw_blocks):
                    current += " " + raw_blocks[i + 1]
                    i += 1

                if self.is_valid_block(current):
                    merged_paragraphs.append(current)
                i += 1

            return merged_paragraphs

        def extract_images_and_ocr(self) -> list[str]:
            """
            Extract images and their associated OCR text using Docling.

            Returns:
                list[str]: List of OCR text strings.
            """
            # Current Docling implementation in this project doesn't focus on raw image extraction here
            return []

        def extract_tables(self) -> list[str]:
            """
            Extract tables from a document using Docling's advanced table-former.

            Returns:
                list[str]: List of tables as Markdown strings.
            """
            result = self.converter.convert(self.file_path)
            tables = []

            for item, _ in result.document.iterate_items():
                if isinstance(item, TableItem):
                    try:
                        # Export the Docling table item to a pandas DataFrame
                        df = item.export_to_dataframe()
                        if not df.empty:
                            tables.append(df.to_markdown(index=False))
                    except Exception:
                        pass

            return tables

        def is_valid_block(self, text: str) -> bool:
            """
            Determine if a text block is valid based on criteria (reused from PDFDistiller).
            """
            if not text:
                return False

            # Must contain at least one alphabetic character (English or Chinese)
            if not re.search(r"[A-Za-z\u4e00-\u9fa5]", text):
                return False

            # Must contain more than two words/terms
            if len(text.split()) <= 2 and len(list(jieba.cut(text))) <= 2:
                return False

            # Exclude common reference/citation patterns
            if self._is_reference_format(text):
                return False

            return True

        def _is_reference_format(self, text: str) -> bool:
            """Helper to identify reference entries."""
            # e.g. [10] Huasheng Liu, ...
            if re.search(r"^\s*\[\s*\d+\s*\]\s*", text):
                return True

            # Keywords common in citations
            reference_keywords = (
                r"(Proceedings|Journal|Conference|pages|vol|ACM|IEEE|arXiv|Springer)"
            )
            if re.search(reference_keywords, text, re.IGNORECASE) and "," in text:
                return True

            return False
else:

    class DoclingDistiller(Distiller):
        def __init__(self, file_path: str):
            raise ImportError(
                "DoclingDistiller requires 'docling' to be installed. "
                "Please install it with `pip install docling`."
            )

        def extract_text_blocks(self) -> list[str]:
            return []

        def extract_images_and_ocr(self) -> list[str]:
            return []

        def extract_tables(self) -> list[str]:
            return []

        def is_valid_block(self, text: str) -> bool:
            return False
