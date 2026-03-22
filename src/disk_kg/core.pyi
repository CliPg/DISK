from typing import Literal

from .distiller import Distiller
from .models import KnowledgeGraph
from .provider import ChatClient, Embeddings

class DISK:
    """
    Domain Incremental conStruction of Knowledge Graphs (DISK).
    一个快速知识图谱的领域增量构建工具
    """
    def __init__(
        self,
        models: ChatClient,
        embeddings: Embeddings,
        language: str = "",
        enable_token_track: bool = True,
    ): ...
    """
    init the DISK instance
    Args:
        models:     Language model insntall
        embeddings: Embeddings instance
        language:   'zh' for Chinese, 'en' for English, or None for auto-detection
        enable_token_track: enable_token_track
    """

    def build_knowledge_graph(
        self,
        file: Distiller,
        batch_size: int = 32,
        max_workers: int | None = None,
        mode: Literal["parallel", "serial"] = "parallel",
    ) -> KnowledgeGraph: ...
