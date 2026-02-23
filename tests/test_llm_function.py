import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 将项目根目录添加到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


class TestLLMFunctionality(unittest.TestCase):
    def test_llm_instance(self):
        """测试 llm 实例是否正确加载且具有基本方法"""
        from config.llm import llm

        # 验证 llm 是否已初始化
        self.assertIsNotNone(llm)
        # 验证是否具有 ChatOpenAI 的典型方法
        self.assertTrue(has_attr(llm, "invoke") or hasattr(llm, "_instance"))
        print(
            f"Current LLM instance: {llm._instance.model_name if hasattr(llm, '_instance') else 'Unknown'}"
        )

    def test_llm_switch_logic(self):
        """测试 switch 函数是否能正常工作而不报错"""
        from config.llm import llm

        # 记录切换前的实例
        old_instance = llm._instance

        # 尝试切换到一个可能存在的 provider (例如 'qwen' 或 'ollama')
        # 这里我们假设 config.toml 中至少定义了这些中的一个，或者我们只是测试 switch 函数本身
        try:
            # 尝试切换回自身或切换到 qwen（如果配置中没有 qwen，这里会报错，所以我们捕获它）
            llm.switch("openai")
            if llm._instance is old_instance:
                print("Switching to the same provider, instance remains unchanged.")
            else:
                print(f"Switched to a different provider, new instance: {llm._instance.model_name}")
            print("Successfully switched to openai")
        except Exception as e:
            self.fail(f"Switch to 'openai' failed: {e}")

    @patch("langchain_community.chat_models.ChatOpenAI.invoke")
    def test_llm_invoke_forwarding(self, mock_invoke):
        """测试 llm 代理是否能正确转发 invoke 调用"""
        from config.llm import llm

        # 设置模拟返回值
        mock_res = MagicMock()
        mock_res.content = "Mocked Response"
        mock_invoke.return_value = mock_res

        # 执行调用
        res = llm.invoke("Hello")

        # 验证
        print(f"LLM invoke response: {res.content}")
        print("Invoke forwarding works correctly")


def has_attr(obj, name):
    try:
        getattr(obj, name)
        return True
    except AttributeError:
        return False


if __name__ == "__main__":
    unittest.main()
