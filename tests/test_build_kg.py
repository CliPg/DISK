import os
import sys
from pathlib import Path
ROOT = Path().resolve().parent
sys.path.append(str(ROOT))

from disk import DISK
from config.llm import llm, embeddings

pdf_path = "tests/sample.pdf"
disk = DISK(llm=llm, embeddings=embeddings)
kg = disk.build_knowledge_graph(pdf_path)

pdf_path = "tests/sample.pdf"
disk = DISK(llm=llm, embeddings=embeddings)
kg = disk.build_knowledge_graph_single_extractor(pdf_path)