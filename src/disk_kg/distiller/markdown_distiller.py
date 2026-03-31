import re

import jieba

from .distiller import Distiller


class MarkdownDistiller(Distiller):
    """
    Distiller for .md (Markdown) files.
    Extracts text blocks, tables, and handles basic Markdown structure.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)

    def extract_text_blocks(self) -> list[str]:
        """
        Extract text blocks (paragraphs) from a Markdown file.
        Blocks shorter than 10 characters are merged with the next block.

        Returns:
            list[str]: List of valid text paragraphs.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.file_path, "r", encoding="gbk") as f:
                content = f.read()

        # Remove markdown tables from text blocks (they will be handled by extract_tables)
        content = re.sub(r"\|.*\|(\r?\n\|.*\|)*", "", content)

        # Remove code blocks
        content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
        content = re.sub(r"`.*?`", "", content)

        raw_blocks = []
        # Split by empty lines or multiple newlines
        for part in re.split(r"\n\s*\n", content):
            text = part.strip()
            if text:
                # Remove markdown headers (#) but keep the content
                text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
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

        # must contain more than two words/terms
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
        # e.g.[10] Huasheng Liu, ...
        pattern1 = r"\s*\[\s*\d+\s*\]\s*[A-Z][a-z]+(\s+([A-Z]\.?|\w+))?"
        # email
        pattern2 = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        # keywords
        reference_keywords = (
            r"(Proceedings|Journal|Conference|pages|vol|ACM|IEEE|arXiv|Springer)"
        )

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
        Extract images and potentially OCR them.
        For Markdown, images are usually external links, so we return empty for now.
        """
        return []

    def extract_tables(self) -> list[str]:
        """
        Extract tables from a Markdown file.
        Returns tables as Markdown strings.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.file_path, "r", encoding="gbk") as f:
                content = f.read()

        # Find markdown tables: lines starting and ending with |
        table_pattern = r"(\|.*\|(?:\r?\n\|.*\|)*)"
        tables = re.findall(table_pattern, content)

        # Filter out lines that are just separators like |---|---|
        valid_tables = []
        for table in tables:
            if re.search(r"[a-zA-Z\u4e00-\u9fa5]", table):
                valid_tables.append(table.strip())

        return valid_tables
