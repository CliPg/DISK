from distiller import PDFDistiller
from extractor import EntitiesExtractor, RelationsExtractor
from manager import KGManager
from models import KnowledgeGraph
from tqdm import tqdm

class DISK:
    """
    Domain Incremental conStruction of Knowledge Graphs (DISK).
    """
    def __init__(self, llm, embeddings, kg:KnowledgeGraph=None):
        self.distiller = PDFDistiller()
        self.entities_extractor = EntitiesExtractor(llm=llm, embeddings=embeddings)
        self.relations_extractor = RelationsExtractor(llm=llm, embeddings=embeddings)
        self.kg_manager = KGManager(kg=kg)

    def build_knowledge_graph(self, pdf_path:str) -> KnowledgeGraph:
        all_entities = []
        all_relations = []

        # Step 1: Distill PDF to text
        texts = self.distiller.extract_text_blocks(pdf_path)

        # Step 2: Extract entities
        print("Extracting entities...")
        for text in tqdm(texts):
            entities = self.entities_extractor.extract_entities(text)
            if entities == None:
                continue
            for entity in entities:
                all_entities.append(entity)
            
        # Step 3: Extract relations
        print("Extracting relations...")
        for text in tqdm(texts):
            relations = self.relations_extractor.extract_relations(text)
            if relation == None:
                continue
            for relation in relations:
                all_relations.append(relation)

        # Step 4: Build Knowledge Graph
        self.kg_manager.add_entities(all_entities)
        self.kg_manager.add_relations(all_relations)

        return self.kg_manager.kg