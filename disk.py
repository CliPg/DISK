import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local, Lock
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


class RateLimiter:
    """速率限制器，控制 API 调用频率."""

    def __init__(self, max_requests_per_second: float = 5):
        """
        Args:
            max_requests_per_second: 每秒最大请求数，默认 5
        """
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_call_time = 0
        self.lock = Lock()

    def acquire(self) -> float:
        """
        获取许可，返回需要等待的时间.

        Returns:
            需要等待的秒数
        """
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time

            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                self.last_call_time = current_time + wait_time
                return wait_time
            else:
                self.last_call_time = current_time
                return 0

    def wait_if_needed(self):
        """如有需要，等待以符合速率限制."""
        wait_time = self.acquire()
        if wait_time > 0:
            time.sleep(wait_time)


def _get_local_extractor(llm, embeddings, language: str = None):
    """获取当前线程的 Extractor 实例."""
    if not hasattr(_thread_local, "extractor"):
        _thread_local.extractor = Extractor(llm=llm, embeddings=embeddings, language=language)
    return _thread_local.extractor


def _extract_with_local_extractor(text, llm, embeddings, rate_limiter: RateLimiter | None = None, pdf_path: str = None, language: str = None):
    """在线程池中执行的提取函数."""
    if rate_limiter is not None:
        rate_limiter.wait_if_needed()
    extractor = _get_local_extractor(llm, embeddings, language)
    return extractor.extract_relations_and_entities(text, pdf_path=pdf_path)


class DISK:
    """
    Domain Incremental conStruction of Knowledge Graphs (DISK).
    """

    def __init__(
        self,
        llm,
        embeddings,
        kg: KnowledgeGraph | None = None,
        max_requests_per_second: float = 5,
        language: str = None,
    ):
        """
        Args:
            llm: Language model instance
            embeddings: Embeddings instance
            kg: Optional existing KnowledgeGraph
            max_requests_per_second: Rate limit for API calls
            language: 'zh' for Chinese, 'en' for English, or None for auto-detection
        """
        self.distiller = PDFDistiller()
        self.language = language
        self.entities_extractor = EntitiesExtractor(llm=llm, embeddings=embeddings, language=language)
        self.relations_extractor = RelationsExtractor(llm=llm, embeddings=embeddings, language=language)
        self.extractor = Extractor(llm=llm, embeddings=embeddings, language=language)
        self.kg_manager = KGManager(kg=kg)
        self.merger = Merger()
        self.llm = llm
        self.embeddings = embeddings
        self.rate_limiter = RateLimiter(max_requests_per_second=max_requests_per_second)
        self.current_pdf_path = None  # Store current pdf_path for language detection

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
        # Store current pdf_path for language detection
        self.current_pdf_path = pdf_path

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

        # Auto-detect language if not set
        if self.language is None and texts:
            from utils.lang_detect import detect_document_language
            from utils.prompts import get_prompts
            detected_lang = detect_document_language(file_path=pdf_path, text_content=texts[0][:500] if texts else "")
            logger.info(f"Detected document language: {detected_lang}")
            # Update extractors with detected language
            prompts = get_prompts(detected_lang)
            self.entities_extractor.prompts = prompts
            self.relations_extractor.prompts = prompts
            self.extractor.prompts = prompts

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
            # 提交所有任务，传入 rate_limiter, pdf_path 和 language
            futures = {
                executor.submit(
                    _extract_with_local_extractor, text, self.llm, self.embeddings, self.rate_limiter, self.current_pdf_path, self.language
                ): i
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
                result = self.extractor.extract_relations_and_entities(text, pdf_path=self.current_pdf_path)
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
        """并发两两归并式合并提取结果."""
        # Step 1: 收集所有非空的结果块
        blocks = []
        for result in results:
            if result is None:
                continue
            relations, entities = result
            if entities and relations:
                blocks.append((relations, entities))

        if not blocks:
            logger.warning("No valid blocks to merge")
            return all_relations, all_entities

        logger.info(f"Starting concurrent pairwise merge with {len(blocks)} blocks")

        # Step 2: 并发两两归并，直到只剩一个块
        round_num = 0
        while len(blocks) > 1:
            round_num += 1
            logger.info(f"Merge round {round_num}: {len(blocks)} blocks -> {(len(blocks) + 1) // 2} blocks")

            # 将块两两分组
            pairs = []
            for i in range(0, len(blocks), 2):
                if i + 1 < len(blocks):
                    pairs.append((blocks[i], blocks[i + 1]))
                else:
                    # 奇数个块时，最后一个直接保留
                    pairs.append((blocks[i], None))

            # 并发合并每一对
            new_blocks = []
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._merge_pair, pair, round_num, idx): idx
                    for idx, pair in enumerate(pairs)
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        if result:
                            new_blocks.append(result)
                    except Exception as e:
                        logger.error(f"Error merging pair {idx} in round {round_num}: {e}")

            # 按原始顺序排序，保持稳定性
            new_blocks.sort(key=lambda x: x[2] if len(x) > 2 else 0)
            # 移除索引标记
            blocks = [block[:2] for block in new_blocks]

        # Step 3: 此时 blocks 只有一个元素，与现有图谱合并
        new_relations, new_entities = blocks[0]

        if len(all_entities) > 0 and len(all_relations) > 0:
            logger.info("Merging new document with existing knowledge graph...")
            all_relations, all_entities = self.merger.merge(
                entities1=all_entities,
                relations1=all_relations,
                entities2=new_entities,
                relations2=new_relations,
            )
        else:
            all_entities = new_entities
            all_relations = new_relations

        logger.info(f"Completed concurrent pairwise merge in {round_num} rounds")
        return all_relations, all_entities

    def _merge_pair(self, pair: tuple, round_num: int, pair_idx: int) -> tuple | None:
        """
        合并一对块。

        Args:
            pair: (block1, block2)，block2可能为None
            round_num: 当前合并轮次
            pair_idx: 当前对的索引

        Returns:
            (merged_relations, merged_entities, original_idx) 或 None
        """
        block1, block2 = pair
        relations1, entities1 = block1

        if block2 is None:
            # 只有一个块，直接返回（带上索引用于排序）
            return (relations1, entities1, pair_idx)

        relations2, entities2 = block2

        try:
            merged_relations, merged_entities = self.merger.merge(
                entities1=entities1,
                relations1=relations1,
                entities2=entities2,
                relations2=relations2,
            )
            logger.debug(f"Round {round_num}, pair {pair_idx}: merged {len(entities1)}+{len(entities2)} entities -> {len(merged_entities)} entities")
            return (merged_relations, merged_entities, pair_idx)
        except Exception as e:
            logger.error(f"Error in round {round_num}, pair {pair_idx}: {e}")
            # 失败时返回第一个块
            return (relations1, entities1, pair_idx)

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
