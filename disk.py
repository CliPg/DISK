from distiller import PDFDistiller
from extractor import EntitiesExtractor, RelationsExtractor
from manager import KGManager
from models import KnowledgeGraph

class DISK:
    """
    Domain Incremental conStruction of Knowledge Graphs (DISK).
    """
    def __init__(self, llm, embeddings):
        self.distiller = PDFDistiller()
        self.entities_extractor = EntitiesExtractor(llm=llm, embeddings=embeddings)
        self.relations_extractor = RelationsExtractor(llm=llm, embeddings=embeddings)
        self.kg_manager = KGManager()

    def build_knowledge_graph(self, pdf_path:str) -> KnowledgeGraph:
        all_entities = []
        all_relations = []

        # Step 1: Distill PDF to text
        texts = self.distiller.extract_text_blocks(pdf_path)

        # Step 2: Extract entities
        for text in texts:
            entities = self.entities_extractor.extract_entities(text)
            for entity in entities:
                if entity == None:
                    continue
                all_entities.append(entity)
            
        # Step 3: Extract relations
        for text in texts:
            relations = self.relations_extractor.extract_entities(text)
            for relation in relations:
                if relation == None:
                    continue
                all_relations.append(relation)

        # Step 4: Build Knowledge Graph
        self.kg_manager.add_entities(all_entities)
        self.kg_manager.add_relations(all_relations)

        return self.kg_manager.kg