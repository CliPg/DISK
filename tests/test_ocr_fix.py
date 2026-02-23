import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from distiller.pdf_distiller import PDFDistiller


def test_extract_images_and_ocr():
    pdf_path = os.path.join(os.path.dirname(__file__), "with_image.pdf")
    if not os.path.exists(pdf_path):
        print(f"Sample PDF not found at {pdf_path}")
        return

    distiller = PDFDistiller()
    print("Running extract_images_and_ocr...")
    results = distiller.extract_images_and_ocr(pdf_path)

    print(f"Extracted {len(results)} images.")
    for i, res in enumerate(results):
        print(f"Image {i + 1}: Page {res['page']}, Text Length: {len(res['ocr_text'])}")
        if len(res["ocr_text"]) > 0:
            print(f"Text Snippet: {res['ocr_text']}")


if __name__ == "__main__":
    test_extract_images_and_ocr()
