# pass at 2026/03/31
import sys

from pydantic import BaseModel, Field

from disk_kg.provider import ChatProxy, Embeddings
from disk_kg.utils.parser import Parser


# 定义一个用于测试的 Pydantic 模型
class UserInfo(BaseModel):
    name: str = Field(description="The name of the user")
    age: int = Field(description="The age of the user")
    hobbies: list[str] = Field(description="List of user hobbies")


def test_parser_with_chat_proxy():
    print("Testing Parser with ChatProxy and Embeddings (Real Environment)...")

    config_path = "config.toml"

    # 初始化真实的 ChatProxy 和 Embeddings (从配置加载)
    llm = ChatProxy(config_path)
    embeddings = Embeddings.build_from(config_path)

    # 初始化 Parser
    parser_instance = Parser(llm=llm, embeddings=embeddings)

    # 调用待测方法
    text = "Charlie is 28 years old and loves swimming and photography."
    prompt = "Extract user information from the text."

    result = parser_instance.extract_information_as_json_from_text(
        text=text, output_structure=UserInfo, prompt=prompt
    )

    # 验证结果
    print(f"Result: {result}")
    assert isinstance(result, dict)
    assert "name" in result
    assert "age" in result
    assert "hobbies" in result

    # 允许 LLM 返回的值有一定的灵活性，但通常应该匹配
    assert result["name"] == "Charlie"
    assert int(result["age"]) == 28
    assert any("swim" in h.lower() for h in result["hobbies"])
    assert any("photograph" in h.lower() for h in result["hobbies"])

    print("✓ Parser with ChatProxy test passed.")


if __name__ == "__main__":
    try:
        test_parser_with_chat_proxy()
        print("\nAll Parser (with ChatProxy) tests passed!")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nTest failed: {e}")
        sys.exit(1)
