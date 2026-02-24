import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
from typing import Literal

from tqdm import tqdm

from distiller import PDFDistiller
from extractor import EntitiesExtractor, Extractor, RelationsExtractor
from manager import KGManager
from merger import Merger
from models import KnowledgeGraph
from models.neo4j_connector import Neo4jConnector
from utils import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

# 线程本地存储，每个线程拥有独立的 Extractor 实例
_thread_local = local()


def _get_local_extractor(llm, embeddings):
    """获取当前线程的 Extractor 实例."""
    if not hasattr(_thread_local, "extractor"):
        _thread_local.extractor = Extractor(llm=llm, embeddings=embeddings)
    return _thread_local.extractor


def _extract_with_local_extractor(text, llm, embeddings):
    """在线程池中执行的提取函数."""
    extractor = _get_local_extractor(llm, embeddings)
    return extractor.extract_relations_and_entities(text)


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
        self.llm = llm
        self.embeddings = embeddings

    """
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
    """

    def build_knowledge_graph(
        self,
        pdf_path: str,
        batch_size: int = 32,
        max_workers: int | None = None,
        mode: Literal["parallel", "serial"] = "parallel",
    ) -> KnowledgeGraph:
        """
        构建知识图谱.

        Args:
            pdf_path: PDF 文件路径
            batch_size: 分批合并的批大小，默认 32
            max_workers: 最大并发线程数，None 时自动计算
            mode: 执行模式，"parallel" 为并行，"serial" 为串行

        Returns:
            KnowledgeGraph: 构建好的知识图谱
        """
        if self.kg_manager.is_existing_kg:
            all_entities = self.kg_manager.kg.entities
            all_relations = self.kg_manager.kg.relations
        else:
            all_entities = []
            all_relations = []

        # 动态计算并发数
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) * 4)

        # Step 1: 提取文本块
        texts = self.distiller.extract_text_blocks(pdf_path)
        logger.info(f"Extracted {len(texts)} text blocks from PDF")

        # Step 2: 提取实体和关系
        if mode == "parallel":
            results = self._extract_parallel(texts, max_workers)
        else:
            results = self._extract_serial(texts)

        # Step 3: 分批合并
        all_relations, all_entities = self._batch_merge(
            results, all_relations, all_entities, batch_size
        )

        # Step 4: 构建知识图谱
        self.kg_manager.add_entities(all_entities)
        self.kg_manager.add_relations(all_relations)

        return self.kg_manager.kg

    def _extract_parallel(self, texts: list[str], max_workers: int) -> list:
        """并行提取实体和关系."""
        results = [None] * len(texts)
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(_extract_with_local_extractor, text, self.llm, self.embeddings): i
                for i, text in enumerate(texts)
            }

            # 收集结果
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting entities and relations (parallel)",
            ):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing block {idx}: {e}")
                    results[idx] = None
                    failed_count += 1

        if failed_count > 0:
            logger.warning(f"Failed to extract {failed_count}/{len(texts)} blocks")

        return results

    def _extract_serial(self, texts: list[str]) -> list:
        """串行提取实体和关系."""
        results = []
        logger.info("Extracting entities and relations (serial)...")

        for text in tqdm(texts, desc="Extracting entities and relations (serial)"):
            try:
                result = self.extractor.extract_relations_and_entities(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text block: {e}")
                results.append(None)

        return results

    def _batch_merge(
        self,
        results: list,
        all_relations: list,
        all_entities: list,
        batch_size: int,
    ) -> tuple[list, list]:
        """分批合并提取结果."""
        batch_entities = []
        batch_relations = []
        merge_count = 0

        for i, result in enumerate(results):
            if result is None:
                continue

            relations, entities = result
            batch_entities.extend(entities)
            batch_relations.extend(relations)

            # 每处理 batch_size 个块进行一次合并
            if (i + 1) % batch_size == 0 or i == len(results) - 1:
                if len(batch_entities) > 0 and len(batch_relations) > 0:
                    if len(all_entities) > 0 and len(all_relations) > 0:
                        logger.info(f"Merging batch {merge_count + 1}...")
                        all_relations, all_entities = self.merger.merge(
                            entities1=all_entities,
                            relations1=all_relations,
                            entities2=batch_entities,
                            relations2=batch_relations,
                        )
                        merge_count += 1
                    else:
                        all_entities = batch_entities
                        all_relations = batch_relations

                batch_entities = []
                batch_relations = []

        logger.info(f"Completed {merge_count} merges")
        return all_relations, all_entities

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
