from distiller import PDFDistiller
from extractor import EntitiesExtractor, RelationsExtractor, Extractor
from manager import KGManager
from models import KnowledgeGraph
from tqdm import tqdm
from utils import load_checkpoint, save_checkpoint

class DISK:
    """
    Domain Incremental conStruction of Knowledge Graphs (DISK).
    """
    def __init__(self, llm, embeddings, kg:KnowledgeGraph=None):
        self.distiller = PDFDistiller()
        self.entities_extractor = EntitiesExtractor(llm=llm, embeddings=embeddings)
        self.relations_extractor = RelationsExtractor(llm=llm, embeddings=embeddings)
        self.extractor = Extractor(llm=llm, embeddings=embeddings)
        self.kg_manager = KGManager(kg=kg)

    def build_knowledge_graph(self, pdf_path:str) -> KnowledgeGraph:
        all_entities = []
        all_relations = []

        ckpt = load_checkpoint()
        start_entity_block = ckpt["entity_block_idx"]
        start_relation_block = ckpt["relation_block_idx"]

        # Step 1: Distill PDF to text
        texts = self.distiller.extract_text_blocks(pdf_path)

        # Step 2: Extract entities
        print("Extracting entities...")
        for i in tqdm(range(start_entity_block, len(texts))):
            text = texts[i]
            entities = self.entities_extractor.extract_entities(text)
            if entities == None:
                continue
            for entity in entities:
                all_entities.append(entity)

            save_checkpoint(pdf_idx=0, entity_block_idx=i+1, relation_block_idx=0)
            
        # Step 3: Extract relations
        print("Extracting relations...")
        for i in tqdm(range(start_relation_block, len(texts))):
            text = texts[i]
            relations = self.relations_extractor.extract_relations(text)
            if relations == None:
                continue
            for relation in relations:
                all_relations.append(relation)

            save_checkpoint(pdf_idx=0, entity_block_idx=len(texts), relation_block_idx=i+1)

        # Step 4: Build Knowledge Graph
        self.kg_manager.add_entities(all_entities)
        self.kg_manager.add_relations(all_relations)

        return self.kg_manager.kg

    def build_knowledge_graph_single_extractor(self, pdf_path:str) -> KnowledgeGraph:
        all_entities = []
        all_relations = []

        # Step 1: Distill PDF to text
        texts = self.distiller.extract_text_blocks(pdf_path)

        # Step 2: Extract entities and relations
        print("Extracting entities and relations...")
        for text in tqdm(texts):
            relations, entities = self.extractor.extract_relations_and_entities(text)
            if relations == None or entities == None:
                continue
            for entity in entities:
                all_entities.append(entity)
            for relation in relations:
                all_relations.append(relation)

        # Step 3: Build Knowledge Graph
        self.kg_manager.add_entities(all_entities)
        self.kg_manager.add_relations(all_relations)

        return self.kg_manager.kg