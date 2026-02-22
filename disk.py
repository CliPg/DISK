from tqdm import tqdm

from distiller import PDFDistiller
from extractor import EntitiesExtractor, Extractor, RelationsExtractor
from manager import KGManager
from merger import Merger
from models import KnowledgeGraph
from models.neo4j_connector import Neo4jConnector
from utils import load_checkpoint, save_checkpoint


class DISK:
    """
    Domain Incremental conStruction of Knowledge Graphs (DISK).
    """

    def __init__(self, llm, embeddings, kg: KnowledgeGraph | None = None):
        self.distiller = PDFDistiller()
        self.entities_extractor = EntitiesExtractor(llm=llm, embeddings=embeddings)
        self.relations_extractor = RelationsExtractor(llm=llm, embeddings=embeddings)
        self.extractor = Extractor(llm=llm, embeddings=embeddings)
        self.kg_manager = KGManager(kg=kg)
        self.merger = Merger()

    def build_knowledge_graph(self, pdf_path: str) -> KnowledgeGraph:
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
            if entities is None:
                continue
            for entity in entities:
                all_entities.append(entity)

            save_checkpoint(pdf_idx=0, entity_block_idx=i + 1, relation_block_idx=0)

        # Step 3: Extract relations
        print("Extracting relations...")
        for i in tqdm(range(start_relation_block, len(texts))):
            text = texts[i]
            relations = self.relations_extractor.extract_relations(text)
            if relations is None:
                continue
            for relation in relations:
                all_relations.append(relation)

            save_checkpoint(pdf_idx=0, entity_block_idx=len(texts), relation_block_idx=i + 1)

        # Step 4: Build Knowledge Graph
        self.kg_manager.add_entities(all_entities)
        self.kg_manager.add_relations(all_relations)

        return self.kg_manager.kg

    def build_knowledge_graph_single_extractor(self, pdf_path: str) -> KnowledgeGraph:
        all_entities = []
        all_relations = []
        # ckpt = load_checkpoint()
        # start_relation_block = ckpt["relation_block_idx"]
        start_relation_block = 0

        # Step 1: Distill PDF to text
        texts = self.distiller.extract_text_blocks(pdf_path)

        # Step 2: Extract entities and relations
        print("Extracting entities and relations...")
        for i in tqdm(range(start_relation_block, len(texts))):
            text = texts[i]
            result = self.extractor.extract_relations_and_entities(text)
            if result is None:
                continue  # 跳过无效 block
            relations, entities = result
            if len(all_entities) > 0 and len(all_relations) > 0:
                print("Merging knowledge graphs...")
                all_relations, all_entities = self.merger.merge(
                    entities1=all_entities,
                    relations1=all_relations,
                    entities2=entities,
                    relations2=relations,
                )
            else:
                all_entities = entities
                all_relations = relations
            # save_checkpoint(pdf_idx=0, entity_block_idx=0, relation_block_idx=i+1)

        # Step 3: Build Knowledge Graph
        self.kg_manager.add_entities(all_entities)
        self.kg_manager.add_relations(all_relations)

        return self.kg_manager.kg

    def visualize_knowledge_graph(
        self, uri, user, password, entities: list | None = None, relations: list | None = None
    ):
        connector = Neo4jConnector(uri=uri, user=user, password=password)
        if entities is not None:
            connector.create_entities(entities)
        else:
            connector.create_entities(self.kg_manager.kg.entities)
        if relations is not None:
            connector.create_relations(relations)
        else:
            connector.create_relations(self.kg_manager.kg.relations)
        connector.close()
