from .distiller import PDFDistiller
from .extractor import EntitiesExtractor, RelationsExtractor

class Disk:
    
    def __init__(self, llm, embeddings):
        self.distiller = PDFDistiller()
        self.entities_extractor = EntitiesExtractor(llm=llm, embeddings=embeddings)
        self.relations_extractor = RelationsExtractor(llm=llm, embeddings=embeddings)