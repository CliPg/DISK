import os
from pathlib import Path

from .chats import ChatProxy, RateLimiter
from .embeddings import Embeddings

# Legacy support for old imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"

# These will be lazily initialized or updated by mcp.py
_llm = None
_embeddings = None


def _get_llm():
    global _llm
    if _llm is None:
        if os.path.exists(CONFIG_PATH):
            _llm = RateLimiter(ChatProxy(CONFIG_PATH))
    return _llm


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        if os.path.exists(CONFIG_PATH):
            _embeddings = Embeddings.build_from(CONFIG_PATH)
    return _embeddings


# Proxy objects or properties could be used here,
# but for simplicity we'll just provide the objects if config exists.
# Note: mcp.py dynamically overwrites CONFIG_PATH, so we need to be careful.


class LegacyLLM:
    def __getattr__(self, name):
        return getattr(_get_llm(), name)

    def __call__(self, *args, **kwargs):
        return _get_llm()(*args, **kwargs)


class LegacyEmbeddings:
    def __getattr__(self, name):
        return getattr(_get_embeddings(), name)


llm = LegacyLLM()
embeddings = LegacyEmbeddings()
