import sys

from disk_kg.provider import ChatProxy, Embeddings, RateLimiter


def test_chat_proxy_initialization():
    print("Testing ChatProxy initialization...")
    config_path = "config.toml"
    proxy = ChatProxy(config_path)

    # Check default model from config (disk.llm = "ollama")
    print(f"Active instance model: {proxy.model_name}")
    assert "qwen3" in proxy.model_name.lower() or "ollama" in str(proxy._instance)
    print("✓ ChatProxy initialized with default provider successfully.")


def test_chat_proxy_switch():
    print("\nTesting ChatProxy switch...")
    config_path = "config.toml"
    proxy = ChatProxy(config_path)

    # Switch to gemini
    proxy.switch("gemini")
    print(f"Switched to: {proxy.model_name}")
    assert "gemini" in proxy.model_name.lower()

    # Switch to gitcode
    proxy.switch("gitcode")
    print(f"Switched to: {proxy.model_name}")
    assert "GLM-5" in proxy.model_name or "zai-org" in proxy.model_name
    print("✓ ChatProxy switched providers successfully.")


def test_rate_limiter_wrapping():
    print("\nTesting RateLimiter wrapping ChatProxy...")
    config_path = "config.toml"
    proxy = ChatProxy(config_path)
    limiter = RateLimiter(proxy, max_request_per_min=10)

    # Test attribute delegation
    print(f"Limiter delegated model_name: {limiter.model_name}")
    assert limiter.model_name == proxy.model_name
    print("✓ RateLimiter delegated attributes correctly.")


def test_embeddings_build():
    print("\nTesting Embeddings build_from...")
    config_path = "config.toml"
    embeddings = Embeddings.build_from(config_path)
    print(f"Embeddings model: {embeddings.model}")
    assert embeddings.model == "text-embedding-v2"
    print("✓ Embeddings built from config successfully.")


if __name__ == "__main__":
    try:
        test_chat_proxy_initialization()
        test_chat_proxy_switch()
        test_rate_limiter_wrapping()
        test_embeddings_build()
        print("\nAll Provider tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
