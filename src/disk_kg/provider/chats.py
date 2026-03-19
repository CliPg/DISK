from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from tomllib import load as load_toml


class ChatClient(Runnable):
    def __init__(self, config: str):
        with open(config, "rb") as fin:
            self.config = load_toml(fin)

    def _get_parmas(self, provider_name):
        cfg = self.config.get("model", {}).get(provider_name, {})
        return {
            "model": cfg.get("model"),
            "api_key": cfg.get("api_key", ""),
            "base_url": cfg.get("api_url"),
        }

    def switch(self, provider_name: str):
        params = self._get_params(provider_name)
        self._instance = ChatOpenAI(temperature=0, max_retries=3, **params)  # type: ignore

    def __getattr__(self, name):
        return getattr(self._instance, name)

    # Runnable interface implementation
    def invoke(self, input, config=None, **kwargs):
        return self._instance.invoke(input, config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        return self._instance.stream(input, config, **kwargs)

    def batch(self, inputs, config=None, **kwargs):
        return self._instance.batch(inputs, config, **kwargs)
