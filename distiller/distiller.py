from abc import ABC, abstractmethod
from pathlib import Path


class Distiller(ABC):
    """
    Abstract base class for document distillers.
    Subclasses must implement all public methods to handle
    specific document formats (PDF, DOCX, etc.).
    """

    def __init__(self):
        # Set log directory to project root/logs
        self.log_dir = Path(__file__).resolve().parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract_text_blocks(self, file_path: str) -> list:
        """
        Extract text blocks (paragraphs) from a document.

        Args:
            file_path (str): Path to the document file.

        Returns:
            list: List of paragraphs extracted from the document.
        """
        raise NotImplementedError("Subclasses must implement extract_text_blocks()")

    @abstractmethod
    def extract_images_and_ocr(self, file_path: str) -> list[dict]:
        """
        Extract images from a document and perform OCR on them.

        Args:
            file_path (str): Path to the document file.

        Returns:
            list[dict]: List of dicts with keys like 'page', 'image', 'ocr_text'.
        """
        raise NotImplementedError("Subclasses must implement extract_images_and_ocr()")

    @abstractmethod
    def extract_tables(self, file_path: str) -> list:
        """
        Extract tables from a document.

        Args:
            file_path (str): Path to the document file.

        Returns:
            list: List of extracted tables (e.g. pandas DataFrames).
        """
        raise NotImplementedError("Subclasses must implement extract_tables()")

    def is_valid_block(self, text: str) -> bool:
        """
        Determine if a text block is valid based on certain criteria.
        Can be overridden by subclasses for format-specific validation.

        Args:
            text (str): The text block to evaluate.

        Returns:
            bool: True if the block is valid, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement is_valid_block()")
