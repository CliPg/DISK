import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
from typing import Literal

from tqdm import tqdm

from .distiller import Distiller
from .extractor import EntitiesExtractor, Extractor, RelationsExtractor
from .models import KnowledgeGraph
from .models.merger import Merger
from .utils import (
    TokenTracker,
    TokenTrackingCallbackHandler,
)
from .utils.lang_detect import detect_document_language
from .utils.prompts import get_prompts

logger = logging.getLogger(__name__)

# 线程本地存储，每个线程拥有独立的 Extractor 实例
_thread_local = local()


def _get_local_extractor(llm, embeddings, language: str = None, token_callback=None):
    """获取当前线程的 Extractor 实例."""
    if not hasattr(_thread_local, "extractor"):
        _thread_local.extractor = Extractor(
            llm=llm, embeddings=embeddings, language=language, token_callback=token_callback
        )
    return _thread_local.extractor


def _extract_with_local_extractor(
    text,
    llm,
    embeddings,
    pdf_path: str = None,
    language: str = None,
    token_callback=None,
):
    """在线程池中执行的提取函数."""
    extractor = _get_local_extractor(llm, embeddings, language, token_callback)
    return extractor.extract_relations_and_entities(text, pdf_path=pdf_path)


class DISK:
    def __init__(
        self,
        model,
        embedding,
        language: str = "",
        enable_token_track: bool = True,
    ):
        """
        Args:
            model: Language model instance
            embedding: Embeddings instance
            language: 'zh' for Chinese, 'en' for English, or None for auto-detection
            enable_token_tracking: 是否启用token使用跟踪
        """
        self.llm = model
        self.embeddings = embedding
        self.language = language

        # Token跟踪
        self.enable_token_tracking = enable_token_track
        self.token_tracker = TokenTracker(model_name="model") if enable_token_track else None
        self.token_callback = (
            TokenTrackingCallbackHandler(self.token_tracker) if enable_token_track else None
        )

        # 使用原始LLM（不包装），通过回调跟踪token
        self.entities_extractor = EntitiesExtractor(
            llm=model, embeddings=embedding, language=language, token_callback=self.token_callback
        )
        self.relations_extractor = RelationsExtractor(
            llm=model, embeddings=embedding, language=language, token_callback=self.token_callback
        )
        self.extractor = Extractor(
            llm=model, embeddings=embedding, language=language, token_callback=self.token_callback
        )
        self.merger = Merger()

    def build_knowledge_graph(
        self,
        file: Distiller,
        batch_size: int = 32,
        max_workers: int | None = None,
        mode: Literal["parallel", "serial"] = "parallel",
        segs: slice | None = None,
    ) -> KnowledgeGraph:
        """
        构建知识图谱.

        Args:
            file: 文件包装器
            batch_size: 分批合并的批大小，默认 32
            max_workers: 最大并发线程数，None 时自动计算
            mode: 执行模式，"parallel" 为并行，"serial" 为串行

        Returns:
            KnowledgeGraph: 构建好的知识图谱
        """
        all_entities = []
        all_relations = []
        self.current_pdf_path = file.file_path

        # 动态计算并发数
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) * 4)

        # Step 1: 提取文本块
        texts = file.extract_text_blocks()
        print(f"Total text blocks extracted: {len(texts)}")
        if segs:
            texts = texts[segs]
        print(f"Extracted {len(texts)} text blocks from PDF")
        logger.info(f"Extracted {len(texts)} text blocks from PDF")

        # Auto-detect language if not set
        if self.language is None and texts:
            detected_lang = detect_document_language(
                file_path=file.file_path, text_content=texts[0][:500] if texts else ""
            )
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
        kg = KnowledgeGraph()
        kg.add_entities(all_entities)
        kg.add_relations(all_relations)

        # 打印Token使用统计
        self.print_token_summary()

        return kg

    def _extract_parallel(self, texts: list[str], max_workers: int) -> list:
        """并行提取实体和关系."""
        results = [None] * len(texts)
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，传入 pdf_path, language 和 token_callback
            futures = {
                executor.submit(
                    _extract_with_local_extractor,
                    text,
                    self.llm,
                    self.embeddings,
                    self.current_pdf_path,
                    self.language,
                    self.token_callback,
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
                result = self.extractor.extract_relations_and_entities(
                    text, pdf_path=self.current_pdf_path
                )
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
            logger.info(
                f"Merge round {round_num}: {len(blocks)} blocks -> {(len(blocks) + 1) // 2} blocks"
            )

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
            logger.debug(
                f"Round {round_num}, pair {pair_idx}: merged {len(entities1)}+{len(entities2)} entities -> {len(merged_entities)} entities"
            )
            return (merged_relations, merged_entities, pair_idx)
        except Exception as e:
            logger.error(f"Error in round {round_num}, pair {pair_idx}: {e}")
            # 失败时返回第一个块
            return (relations1, entities1, pair_idx)

    # --------------------#
    # Token tracking methods #
    # --------------------#

    def print_token_summary(self):
        """打印Token使用统计摘要"""
        if self.token_tracker:
            self.token_tracker.print_summary()
        else:
            print("Token tracking is not enabled.")

    def get_token_summary(self) -> dict | None:
        """获取Token使用统计"""
        if self.token_tracker:
            return self.token_tracker.get_summary()
        return None

    def save_token_usage(self):
        """保存Token使用记录"""
        if self.token_tracker:
            self.token_tracker.save()
