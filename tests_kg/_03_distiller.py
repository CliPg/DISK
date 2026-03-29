import os
import sys

from disk_kg.distiller import Distiller, PDFDistiller


def test_distiller_interface():
    print("=== Testing Distiller Interface ===")

    # 准备测试文件路径
    pdf_path = "tests/sample.pdf"
    docx_path = "tests/document_test.docx"

    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"Warning: {pdf_path} not found, skipping PDF test.")
    else:
        print(f"\n[1] Testing Distiller.distill('{pdf_path}')...")
        # 测试静态方法自动识别和提取
        blocks = Distiller.distill(pdf_path)
        print(f"Success! Extracted {len(blocks)} blocks.")
        assert isinstance(blocks, list)
        if blocks:
            assert isinstance(blocks[0], str)

    if not os.path.exists(docx_path):
        print(f"Warning: {docx_path} not found, skipping DOCX test.")
    else:
        print(f"\n[2] Testing Distiller.distill('{docx_path}')...")
        # 测试静态方法自动识别和提取
        blocks = Distiller.distill(docx_path)
        print(f"Success! Extracted {len(blocks)} blocks.")
        assert isinstance(blocks, list)
        if blocks:
            assert isinstance(blocks[0], str)

    # 测试显式实例化
    if os.path.exists(pdf_path):
        print("\n[3] Testing PDFDistiller instantiation...")
        pd = PDFDistiller(pdf_path)
        blocks = pd.distill()
        print(f"Success! pd.distill() extracted {len(blocks)} blocks.")

        # 测试不传参数调用
        blocks_again = pd.extract_text_blocks()
        assert len(blocks) == len(blocks_again)
        print("Success! pd.extract_text_blocks() without arguments worked.")

    print("\n=== All Interface Tests Passed! ===")


if __name__ == "__main__":
    try:
        test_distiller_interface()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
