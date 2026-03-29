from abc import ABC, abstractmethod
from pathlib import Path


class TextBlock:
    """
    Represents a block of text extracted from a document.
    Contains the text content and its position in the document.
    """

    def __init__(self, text: str, page_number: int, block_id: int, file_hash: str | None = None):
        self.text = text
        self.page_number = page_number
        self.block_id = block_id
        self.file_hash = file_hash

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "page_number": self.page_number,
            "block_id": self.block_id,
            "file_hash": self.file_hash,
        }


class Distiller(ABC):
    """
    Abstract base class for document distillers.
    Subclasses must implement all public methods to handle
    specific document formats (PDF, DOCX, etc.).
    """

    def __init__(self, file_path: str):
        # Set log directory to project root/logs
        self.file_path = file_path
        self.log_dir = Path(__file__).resolve().parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract_text_blocks(self, file_path: str | None = None) -> list[str]:
        """
        Extract text blocks (paragraphs) from a document.

        Args:
            file_path (str, optional): Path to the document file. Defaults to self.file_path.

        Returns:
            list: List of paragraphs extracted from the document.
        """
        raise NotImplementedError("Subclasses must implement extract_text_blocks()")

    @abstractmethod
    def extract_images_and_ocr(self, file_path: str | None = None) -> list[dict]:
        """
        Extract images from a document and perform OCR on them.

        Args:
            file_path (str, optional): Path to the document file. Defaults to self.file_path.

        Returns:
            list[dict]: List of dicts with keys like 'page', 'image', 'ocr_text'.
        """
        raise NotImplementedError("Subclasses must implement extract_images_and_ocr()")

    @abstractmethod
    def extract_tables(self, file_path: str | None = None) -> list:
        """
        Extract tables from a document.

        Args:
            file_path (str, optional): Path to the document file. Defaults to self.file_path.

        Returns:
            list: List of extracted tables (e.g. pandas DataFrames).
        """
        raise NotImplementedError("Subclasses must implement extract_tables()")

    @staticmethod
    def distill(file_path: str) -> list[str]:
        """
        Automatically selects the appropriate distiller based on the file extension
        and extracts text blocks.

        Args:
            file_path (str): Path to the document file.

        Returns:
            list[str]: List of paragraphs extracted from the document.
        """
        from .docling_distiller import HAS_DOCLING, DoclingDistiller
        from .docx_distiller import DocxDistiller
        from .pdf_distiller import PDFDistiller

        ext = Path(file_path).suffix.lower()

        if ext == ".pdf" and HAS_DOCLING:
            # For PDF, use PDFDistiller or DoclingDistiller
            # Default to PDFDistiller for now as it's more specific in current implementation
            distiller = DoclingDistiller(file_path)
        elif ext == ".pdf":
            distiller = PDFDistiller(file_path)
        elif ext in [".docx", ".doc"]:
            distiller = DocxDistiller(file_path)
        else:
            # Fallback to DoclingDistiller for other supported formats
            raise ValueError(f"Unsupported file format: {ext}")

        return distiller.distill()

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
