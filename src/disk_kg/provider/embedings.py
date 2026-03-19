from langchain_openai import OpenAIEmbeddings
from pydantic.types import SecretStr
from tomllib import load as load_toml


class Embeddings(OpenAIEmbeddings):
    def __init__(self, *, model: str, api_key: SecretStr, base_url: str):
        OpenAIEmbeddings.__init__(
            self,
            model=model,
            api_key=api_key,
            base_url=base_url,
            check_embedding_ctx_length=False,
        )

    @staticmethod
    def build_from(config: str):
        with open(config, "rb") as fin:
            _config = load_toml(fin)
        _emb_cfg = _config.get("disk", {}).get("embeddings", {})
        return Embeddings(
            model=_emb_cfg.get("model"),
            api_key=_emb_cfg.get("api_key"),
            base_url=_emb_cfg.get("api_url"),
        )
