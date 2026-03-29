from typing import Any

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

class ChatProxy(Runnable):
    """A proxy class that manages multiple ChatOpenAI model providers.

    It reads configuration from a TOML file and allows switching between different
    LLM providers dynamically.
    """

    config: dict[str, Any]
    models: dict[str, Any]
    _instance: ChatOpenAI

    def __init__(self, config: str) -> None:
        """Initialize the ChatProxy with a configuration file path.

        Args:
            config: Path to the TOML configuration file.
        """
        ...

    def _get_params(self, provider_name: str) -> dict[str, Any]:
        """Extract connection parameters for a specific provider.

        Args:
            provider_name: The name of the provider in the config file.

        Returns:
            A dictionary containing model, api_key, and base_url.
        """
        ...

    def switch(self, provider_name: str) -> None:
        """Switch the active ChatOpenAI instance to the specified provider.

        Args:
            provider_name: The name of the provider to switch to.
        """
        ...

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the internal ChatOpenAI instance."""
        ...

    def invoke(self, input: Any, config: Any = ..., **kwargs: Any) -> Any:
        """Invoke the active ChatOpenAI model."""
        ...

    def stream(self, input: Any, config: Any = ..., **kwargs: Any) -> Any:
        """Stream the output from the active ChatOpenAI model."""
        ...

    def batch(self, inputs: list[Any], config: Any = ..., **kwargs: Any) -> list[Any]:
        """Batch process multiple inputs using the active ChatOpenAI model."""
        ...

class RateLimiter(Runnable):
    """A wrapper class that limits the request rate of a ChatProxy or ChatOpenAI instance.

    Ensures that requests do not exceed a specified number per minute.
    """

    _cls: ChatProxy | ChatOpenAI
    max_request_per_min: int
    min_interval: float
    last_request_time: float

    def __init__(self, cls: ChatProxy | ChatOpenAI, max_request_per_min: int = 60) -> None:
        """Initialize the RateLimiter.

        Args:
            cls: The ChatProxy or ChatOpenAI instance to wrap.
            max_request_per_min: Maximum number of requests allowed per minute.
        """
        ...

    def _wait(self) -> None:
        """Wait if necessary to comply with the rate limit."""
        ...

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapping callables with rate limiting."""
        ...

    def invoke(self, input: Any, config: Any = ..., **kwargs: Any) -> Any:
        """Invoke the active ChatOpenAI model with rate limiting."""
        ...

    def stream(self, input: Any, config: Any = ..., **kwargs: Any) -> Any:
        """Stream the output from the active ChatOpenAI model with rate limiting."""
        ...

    def batch(self, inputs: list[Any], config: Any = ..., **kwargs: Any) -> list[Any]:
        """Batch process multiple inputs using the active ChatOpenAI model with rate limiting."""
        ...

ChatClient = RateLimiter | ChatProxy | ChatOpenAI
