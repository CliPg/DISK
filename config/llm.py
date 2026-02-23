from pathlib import Path

import tomllib
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 路径设置与配置加载
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Config file not found: {CONFIG_PATH}. Please create it from config.example.toml."
    )

with open(CONFIG_PATH, "rb") as f:
    _CONFIG = tomllib.load(f)


def _get_params(provider: str) -> dict[str, str]:
    """从配置中提取 OpenAI 兼容参数"""
    cfg = _CONFIG.get("model", {}).get(provider, {})
    return {
        "model": cfg.get("model"),
        "api_key": cfg.get("api_key", ""),
        "base_url": cfg.get("api_url"),
    }


class LLMProxy:
    """LLM 代理，允许运行时切换底层提供商"""

    def __init__(self, provider: str):
        self.switch(provider)

    def switch(self, provider: str):
        params = _get_params(provider)
        self._instance = ChatOpenAI(temperature=0, max_retries=3, **params)  # type: ignore

    def __getattr__(self, name):
        return getattr(self._instance, name)


# 初始化全局对象
_disk = _CONFIG.get("disk", {})

# LLM 代理初始化
llm = LLMProxy(_disk.get("llm", "openai"))

# Embeddings 直接从 [disk.embeddings] 初始化
_emb_cfg = _disk.get("embeddings", {})
embeddings = OpenAIEmbeddings(
    model=_emb_cfg.get("model"),
    api_key=_emb_cfg.get("api_key"),
    base_url=_emb_cfg.get("api_url"),
    check_embedding_ctx_length=False,
)
