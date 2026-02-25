import os
import sys
from pathlib import Path
ROOT = Path().resolve().parent
sys.path.append(str(ROOT))

from disk import DISK
from config.llm import llm, embeddings

pdf_path = "tests/李明1.pdf"
disk = DISK(llm=llm, embeddings=embeddings)
kg = disk.build_knowledge_graph(pdf_path)


for entity in kg.entities:
    print(f"Entity: {entity.name}, Description: {entity.description}")

for relation in kg.relations:
    print(f"Relation: {relation.name}, Description: {relation.description}")