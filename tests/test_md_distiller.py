from src.disk_kg.distiller.distiller import Distiller
import os

def test_markdown_distiller():
    # Ensure the test file exists
    file_path = "tests/test_markdown.md"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Use factory method to get the correct distiller
    distiller = Distiller.distill(file_path)
    
    print(f"Distiller type: {type(distiller)}")
    assert "MarkdownDistiller" in str(type(distiller)), "Should use MarkdownDistiller"

    # Test Text Block Extraction
    print("\n--- Text Blocks ---")
    blocks = distiller.extract_text_blocks()
    for i, block in enumerate(blocks):
        print(f"Block {i}: {block}")
    
    # Check if we got at least some blocks
    assert len(blocks) > 0, "Should extract at least one text block"
    # Check if headers (without #) are correctly merged or extracted
    assert any("Subheader" in b for b in blocks), "Should contain 'Subheader'"

    # Test Table Extraction
    print("\n--- Tables ---")
    tables = distiller.extract_tables()
    for i, table in enumerate(tables):
        print(f"Table {i}:\n{table}")
    
    assert len(tables) == 1, "Should extract exactly one table"
    assert "Header 1" in tables[0], "Table should contain its header"

    print("\nMarkdown Distiller test passed successfully!")

if __name__ == "__main__":
    test_markdown_distiller()
