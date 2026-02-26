"""
Manual test for DocxDistiller without pytest dependency.
Reads tests/document_test.docx and prints extraction results.
"""

import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from distiller.docx_distiller import DocxDistiller


def test_docx_distiller():
    docx_path = os.path.join(os.path.dirname(__file__), "document_test.docx")
    if not os.path.exists(docx_path):
        print(f"Sample DOCX not found at {docx_path}")
        return

    distiller = DocxDistiller()
    print("=== DOCX Distiller Test ===")
    print(f"File: {docx_path}")

    # 1. Test Text Blocks
    print("\n[1] Testing extract_text_blocks...")
    blocks = distiller.extract_text_blocks(docx_path)
    print(f"Extracted {len(blocks)} blocks.")
    for i, block in enumerate(blocks[:5]):  # Show first 5
        print(f"  Block {i}: {block[:100]}...")
    if len(blocks) > 5:
        print(f"  ... and {len(blocks) - 5} more.")

    # 2. Test Full Text (docx2txt)
    print("\n[2] Testing extract_full_text...")
    full_text = distiller.extract_full_text(docx_path)
    print(f"Full text length: {len(full_text)} characters.")
    print(f"Snippet: {full_text[:200]}...")

    # 3. Test Images and OCR
    print("\n[3] Testing extract_images_and_ocr...")
    images = distiller.extract_images_and_ocr(docx_path)
    print(f"Extracted {len(images)} images.")
    for i, img_data in enumerate(images):
        print(
            f"  Image {i}: Index {img_data['index']}, OCR Text Length: {len(img_data['ocr_text'])}"
        )
        if img_data["ocr_text"]:
            print(f"    OCR Snippet: {img_data['ocr_text'][:100]}...")

    # 4. Test Tables
    print("\n[4] Testing extract_tables...")
    tables = distiller.extract_tables(docx_path)
    print(f"Extracted {len(tables)} tables.")
    for i, table in enumerate(tables):
        print(f"  Table {i}: {len(table)} rows, {len(table[0]) if table else 0} columns")
        for row in table[:2]:  # Show first 2 rows
            print(f"    Row: {row}")
        if len(table) > 2:
            print(f"    ... and {len(table) - 2} more rows.")

    print("\n=== Test Finished ===")


if __name__ == "__main__":
    test_docx_distiller()
