from disk_kg import DISK
from disk_kg.provider.llm import embeddings, llm

pdf_path = "tests/sample.pdf"
disk = DISK(llm=llm, embeddings=embeddings)
kg = disk.build_knowledge_graph(pdf_path)

# pdf_path = "tests/李明1.pdf"
# disk = DISK(llm=llm, embeddings=embeddings)
# kg = disk.build_knowledge_graph(pdf_path)

# pdf_path = "tests/李明2.pdf"
# kg = disk.build_knowledge_graph(pdf_path)
