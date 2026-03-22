from time import sleep, time

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from tomllib import load as load_toml


class ChatProxy(Runnable):
    def __init__(self, config: str):
        with open(config, "rb") as fin:
            self.config = load_toml(fin)

        self.models = self.config.get("model", {})
        if not self.models:
            raise ValueError("No models configured in the config file.")

        default_provider = self.config.get("disk", {}).get("llm", None)
        if default_provider:
            self.switch(default_provider)
        else:
            # If no default provider is specified, use the first one in the config
            self.switch(list(self.models.keys())[0])

    def _get_params(self, provider_name):
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


class RateLimiter:
    def __init__(self, cls: ChatProxy | ChatOpenAI, max_request_per_min: int = 60):
        self._cls = cls
        self.max_request_per_min = max_request_per_min
        self.min_interval = 60 / max_request_per_min
        self.last_request_time = 0

    def _wait(self) -> None:
        elapsed_time = time() - self.last_request_time
        if elapsed_time < self.min_interval:
            sleep(self.min_interval - elapsed_time)

    def __getattr__(self, name):
        attr = getattr(self._cls, name)
        if callable(attr):

            def wrapper(*args, **kwargs):
                self._wait()
                result = attr(*args, **kwargs)
                self.last_request_time = time()
                return result

            return wrapper
        else:
            return attr


ChatClient = RateLimiter | ChatProxy | ChatOpenAI
