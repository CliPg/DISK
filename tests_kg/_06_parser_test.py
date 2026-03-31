import sys
from typing import List
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from disk_kg.utils.parser import Parser
from disk_kg.provider import ChatProxy, Embeddings


# 定义一个用于测试的 Pydantic 模型
class UserInfo(BaseModel):
    name: str = Field(description="The name of the user")
    age: int = Field(description="The age of the user")
    hobbies: List[str] = Field(description="List of user hobbies")


def test_parser_with_chat_proxy():
    print("Testing Parser with ChatProxy and Embeddings...")

    config_path = "config.toml"
    
    # 初始化真实的 ChatProxy 和 Embeddings (从配置加载)
    llm = ChatProxy(config_path)
    embeddings = Embeddings.build_from(config_path)

    # 为了使测试不依赖真实的 LLM 接口返回，我们 mock 内部的 invoke 方法
    # 这样既测试了 Parser 与 ChatProxy 的集成，又保证了测试的可重复性
    json_response = '{"name": "Charlie", "age": 28, "hobbies": ["swimming", "photography"]}'
    mock_message = AIMessage(content=json_response)
    
    # 记录原始方法以便验证或恢复 (如果需要)
    original_invoke = llm.invoke
    llm.invoke = MagicMock(return_value=mock_message)

    # 初始化 Parser
    parser_instance = Parser(llm=llm, embeddings=embeddings)

    # 调用待测方法
    text = "Charlie is 28 years old and loves swimming and photography."
    prompt = "Extract user information from the text."
    
    result = parser_instance.extract_information_as_json_from_text(
        text=text,
        output_structure=UserInfo,
        prompt=prompt
    )

    # 验证结果
    print(f"Result: {result}")
    assert isinstance(result, dict)
    assert result["name"] == "Charlie"
    assert result["age"] == 28
    assert "swimming" in result["hobbies"]
    assert "photography" in result["hobbies"]

    # 验证 llm.invoke 是否被正确调用
    llm.invoke.assert_called_once()
    
    print("✓ Parser with ChatProxy test passed.")


def test_parser_with_token_callback_and_proxy():
    print("\nTesting Parser with token_callback and ChatProxy...")

    config_path = "config.toml"
    llm = ChatProxy(config_path)
    embeddings = Embeddings.build_from(config_path)
    
    mock_callback = MagicMock()
    json_response = '{"name": "Diana", "age": 32, "hobbies": ["hiking"]}'
    llm.invoke = MagicMock(return_value=AIMessage(content=json_response))

    # 初始化带回调的 Parser
    parser_instance = Parser(llm=llm, embeddings=embeddings, token_callback=mock_callback)

    text = "Diana is 32 and likes hiking."
    prompt = "Extract info."
    
    result = parser_instance.extract_information_as_json_from_text(
        text=text,
        output_structure=UserInfo,
        prompt=prompt
    )

    assert result["name"] == "Diana"
    
    # 验证 invoke 时是否传递了 callback
    llm.invoke.assert_called_once()
    args, kwargs = llm.invoke.call_args
    # 在 RunnableSequence 中，llm.invoke(input, config=config)
    # 可能会作为位置参数传递
    config = None
    if "config" in kwargs:
        config = kwargs["config"]
    elif len(args) > 1:
        config = args[1]
    
    assert config is not None, f"Config not found in args {args} or kwargs {kwargs}"
    assert "callbacks" in config
    
    callbacks = config["callbacks"]
    if hasattr(callbacks, "handlers"):
        assert mock_callback in callbacks.handlers
    else:
        assert mock_callback in callbacks

    print("✓ Parser with token_callback and ChatProxy test passed.")


if __name__ == "__main__":
    try:
        test_parser_with_chat_proxy()
        test_parser_with_token_callback_and_proxy()
        print("\nAll Parser (with ChatProxy) tests passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTest failed: {e}")
        sys.exit(1)
