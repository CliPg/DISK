"""
Language detection utility for document processing.
Supports detection from filename and text content.
"""

import re
from pathlib import Path
from typing import Optional

# Try to import fasttext_langdetect, fallback to regex-based detection
try:
    from fasttext_langdetect import detect as ft_detect
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False


def detect_with_fasttext(text: str) -> Optional[str]:
    """
    Detect language using fasttext_langdetect.

    Args:
        text: Text content to analyze

    Returns:
        'zh' for Chinese, 'en' for English, or None
    """
    if not FASTTEXT_AVAILABLE or not text:
        return None

    try:
        result = ft_detect(text[:1000])  # Sample first 1000 chars
        if result and len(result) > 0:
            lang = result[0].get('lang', '') if isinstance(result[0], dict) else str(result[0])
            if lang.startswith('zh'):
                return 'zh'
            elif lang.startswith('en'):
                return 'en'
    except Exception:
        pass

    return None


def detect_from_filename(filename: str) -> Optional[str]:
    """
    Detect document language from filename.

    Args:
        filename: Name of the document file

    Returns:
        'zh' for Chinese, 'en' for English, or None if undetermined
    """
    # Check for Chinese characters in filename
    if re.search(r'[\u4e00-\u9fa5]', filename):
        return 'zh'

    # Check for common Chinese filename patterns
    chinese_keywords = [
        '中文', '文档', '报告', '论文', '章节', '附录',
        'chinese', 'zhongwen', 'zh'
    ]
    filename_lower = filename.lower()
    for keyword in chinese_keywords:
        if keyword in filename_lower:
            return 'zh'

    # Default to English if no Chinese detected
    return 'en'


def detect_from_text(text: str, sample_size: int = 500) -> Optional[str]:
    """
    Detect document language from text content using character analysis.

    Args:
        text: Text content to analyze
        sample_size: Number of characters to sample for detection

    Returns:
        'zh' for Chinese, 'en' for English, or None if undetermined
    """
    if not text:
        return None

    # Try fasttext first if available
    if FASTTEXT_AVAILABLE:
        ft_result = detect_with_fasttext(text)
        if ft_result:
            return ft_result

    # Fallback to regex-based detection
    sample = text[:sample_size] if len(text) > sample_size else text

    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', sample))

    # Count English letters
    english_chars = len(re.findall(r'[a-zA-Z]', sample))

    # Count total valid characters
    total_valid = chinese_chars + english_chars

    if total_valid == 0:
        return None

    # If Chinese characters make up more than 20% of valid characters, consider it Chinese
    chinese_ratio = chinese_chars / total_valid
    if chinese_ratio > 0.2:
        return 'zh'

    return 'en'


def detect_document_language(
    file_path: str = None,
    filename: str = None,
    text_content: str = None,
    use_fasttext: bool = True
) -> str:
    """
    Detect document language using multiple methods.

    Priority: text content > filename > default to English

    Args:
        file_path: Path to the document file
        filename: Name of the document file (alternative to file_path)
        text_content: Sample text content from the document
        use_fasttext: Whether to try using fasttext_langdetect (falls back to regex if unavailable)

    Returns:
        'zh' for Chinese, 'en' for English
    """
    # Extract filename from file_path if needed
    if file_path and not filename:
        filename = Path(file_path).name

    # Method 1: Detect from text content (most reliable)
    if text_content:
        lang = detect_from_text(text_content)
        if lang:
            return lang

    # Method 2: Detect from filename
    if filename:
        lang = detect_from_filename(filename)
        if lang:
            return lang

    # Default to English
    return 'en'
