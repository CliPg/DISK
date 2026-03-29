import os
import sys

# 将 src 目录添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from disk_kg.provider import ChatProxy, Embeddings, RateLimiter

def test_llm_network():
    print("Testing LLM network connectivity with default provider...")
    config_path = "config.toml"
    # 使用 RateLimiter 包装 ChatProxy，模拟真实使用场景
    llm = RateLimiter(ChatProxy(config_path), max_request_per_min=60)
    
    print(f"Using default LLM: {llm.model_name}")
    try:
        # 执行一个简单的 invoke 调用
        response = llm.invoke("Please reply with the word 'OK' if you can hear me.")
        content = response.content.strip()
        print(f"LLM Response: {content}")
        assert len(content) > 0
        print("✓ LLM network call successful.")
    except Exception as e:
        print(f"✗ LLM network call failed: {e}")
        raise

def test_embeddings_network():
    print("\nTesting Embeddings network connectivity with default provider...")
    config_path = "config.toml"
    embeddings = Embeddings.build_from(config_path)
    
    print(f"Using default Embeddings model: {embeddings.model}")
    try:
        # 执行向量化请求
        query = "Hello world"
        vector = embeddings.embed_query(query)
        print(f"Embeddings vector dimension: {len(vector)}")
        assert len(vector) > 0
        print("✓ Embeddings network call successful.")
    except Exception as e:
        print(f"✗ Embeddings network call failed: {e}")
        raise

if __name__ == "__main__":
    try:
        test_llm_network()
        test_embeddings_network()
        print("\nNetwork functionality verified successfully!")
    except Exception as e:
        print(f"\nNetwork test failed: {e}")
        sys.exit(1)
