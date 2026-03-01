"""
Token使用跟踪模块
用于记录LLM调用的输入和输出token数量，支持成本估算。
"""

import json
import os
import time
from datetime import datetime
from threading import Lock
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenTracker:
    """Token使用跟踪器"""

    # 不同模型的大致价格（美元/1K tokens）
    PRICING = {
        # OpenAI models
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},

        # DashScope / Qwen models (估算)
        "qwen-turbo": {"input": 0.0008, "output": 0.0008},
        "qwen-plus": {"input": 0.004, "output": 0.004},
        "qwen-max": {"input": 0.02, "output": 0.02},

        # Ollama / 本地模型（免费）
        "ollama": {"input": 0, "output": 0},
    }

    def __init__(self, log_dir: str = None, model_name: str = "unknown"):
        """
        初始化Token跟踪器

        Args:
            log_dir: 日志目录，默认为项目根目录/logs
            model_name: 模型名称，用于成本估算
        """
        if log_dir is None:
            self.log_dir = Path(__file__).resolve().parent.parent / "logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.records = []
        self.lock = Lock()
        self.session_start = time.time()

        # 获取价格信息
        self.pricing = self._get_pricing(model_name)

        # 日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"token_usage_{timestamp}.log"
        self.json_file = self.log_dir / f"token_usage_{timestamp}.json"

    def _get_pricing(self, model_name: str) -> dict:
        """获取模型价格信息"""
        model_lower = model_name.lower()
        for key, pricing in self.PRICING.items():
            if key in model_lower:
                return pricing
        return {"input": 0, "output": 0}  # 默认免费

    def log(self, input_tokens: int, output_tokens: int,
            prompt: str = None, response: str = None,
            metadata: dict = None):
        """
        记录一次LLM调用的token使用情况

        Args:
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            prompt: 输入提示词（可选）
            response: 输出响应（可选）
            metadata: 额外的元数据（可选）
        """
        with self.lock:
            record = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_input": input_tokens * self.pricing["input"] / 1000,
                "cost_output": output_tokens * self.pricing["output"] / 1000,
                "cost_total": (input_tokens * self.pricing["input"] + output_tokens * self.pricing["output"]) / 1000,
            }

            if prompt:
                record["prompt_preview"] = prompt[:200] + "..." if len(prompt) > 200 else prompt

            if response:
                record["response_preview"] = response[:200] + "..." if len(response) > 200 else response

            if metadata:
                record["metadata"] = metadata

            self.records.append(record)

            # 写入日志文件
            self._write_log(record)
            self._write_json()

    def _write_log(self, record: dict):
        """写入可读日志"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{record['timestamp']}] ")
            f.write(f"Model: {record['model']} | ")
            f.write(f"Input: {record['input_tokens']} | ")
            f.write(f"Output: {record['output_tokens']} | ")
            f.write(f"Total: {record['total_tokens']} | ")
            f.write(f"Cost: ${record['cost_total']:.6f}\n")
            if "prompt_preview" in record:
                f.write(f"  Prompt: {record['prompt_preview']}\n")

    def _write_json(self):
        """写入JSON格式的完整记录"""
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.model_name,
                "session_start": datetime.fromtimestamp(self.session_start).isoformat(),
                "records": self.records,
                "summary": self.get_summary()
            }, f, ensure_ascii=False, indent=2)

    def get_summary(self) -> dict:
        """获取使用统计摘要"""
        total_input = sum(r["input_tokens"] for r in self.records)
        total_output = sum(r["output_tokens"] for r in self.records)
        total_cost = sum(r["cost_total"] for r in self.records)

        return {
            "total_calls": len(self.records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": total_cost,
            "avg_tokens_per_call": (total_input + total_output) / len(self.records) if self.records else 0,
        }

    def print_summary(self):
        """打印使用统计摘要"""
        summary = self.get_summary()
        duration = time.time() - self.session_start

        print("\n" + "=" * 50)
        print("Token使用统计")
        print("=" * 50)
        print(f"模型: {self.model_name}")
        print(f"总调用次数: {summary['total_calls']}")
        print(f"输入Token: {summary['total_input_tokens']:,}")
        print(f"输出Token: {summary['total_output_tokens']:,}")
        print(f"总Token: {summary['total_tokens']:,}")
        print(f"总成本: ${summary['total_cost_usd']:.4f}")
        print(f"平均每次调用: {summary['avg_tokens_per_call']:.0f} tokens")
        print(f"耗时: {duration:.1f}秒")
        print("=" * 50 + "\n")

    def save(self):
        """保存记录到文件"""
        self._write_json()


class TokenTrackingCallbackHandler(BaseCallbackHandler):
    """
    LangChain回调处理器，用于自动跟踪LLM调用的token使用

    用法:
        tracker = TokenTracker(model_name="qwen-turbo")
        callback_handler = TokenTrackingCallbackHandler(tracker)
        # 在invoke时传入callbacks参数
        result = chain.invoke(input, config={"callbacks": [callback_handler]})
    """

    def __init__(self, tracker: TokenTracker):
        super().__init__()
        self.tracker = tracker
        self._last_input = None
        self._call_count = 0

    def on_llm_start(self, prompts: List[str], **kwargs):
        """LLM调用开始时记录输入"""
        self._call_count += 1
        self._last_input = prompts[0] if prompts else None
        # Debug: 打印以确认回调被调用
        print(f"[DEBUG] on_llm_start called (call #{self._call_count})")

    def on_llm_end(self, response, **kwargs):
        """LLM调用结束时记录token使用"""
        print(f"[DEBUG] on_llm_end called, response type: {type(response)}")

        input_tokens = 0
        output_tokens = 0

        # 方法1: 从 response.llm_output 获取 (老版本 LangChain)
        if hasattr(response, 'llm_output') and response.llm_output:
            llm_output = response.llm_output
            if isinstance(llm_output, dict):
                token_usage = llm_output.get('token_usage', {})
                if isinstance(token_usage, dict):
                    input_tokens = token_usage.get('prompt_tokens', 0) or token_usage.get('input_tokens', 0)
                    output_tokens = token_usage.get('completion_tokens', 0) or token_usage.get('output_tokens', 0)
                    print(f"[DEBUG] Got tokens from llm_output.token_usage: {input_tokens}+{output_tokens}")

        # 方法2: 从 response.usage_metadata 获取 (新版本 LangChain)
        if input_tokens == 0 and output_tokens == 0:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.get('input_tokens', 0)
                output_tokens = response.usage_metadata.get('output_tokens', 0)
                print(f"[DEBUG] Got tokens from usage_metadata: {input_tokens}+{output_tokens}")

        # 方法3: 尝试从 generations 中获取
        if input_tokens == 0 and output_tokens == 0 and hasattr(response, 'generations'):
            for gen_list in response.generations:
                for generation in gen_list:
                    if hasattr(generation, 'generation_info'):
                        info = generation.generation_info or {}
                        if isinstance(info, dict):
                            input_tokens = info.get('prompt_tokens', 0) or input_tokens
                            output_tokens = info.get('completion_tokens', 0) or output_tokens
                            print(f"[DEBUG] Got tokens from generation_info: {input_tokens}+{output_tokens}")

        # 方法4: 从 response 直接获取 (某些 provider)
        if input_tokens == 0 and output_tokens == 0:
            if hasattr(response, 'token_usage'):
                token_usage = response.token_usage
                if hasattr(token_usage, 'prompt_tokens'):
                    input_tokens = token_usage.prompt_tokens
                    output_tokens = token_usage.completion_tokens
                    print(f"[DEBUG] Got tokens from response.token_usage: {input_tokens}+{output_tokens}")

        # 如果仍然没有，基于文本长度估算
        if input_tokens == 0 and output_tokens == 0:
            # 估算输入
            if self._last_input:
                input_tokens = len(self._last_input) // 2
            else:
                input_tokens = 100

            # 估算输出
            output_text = ""
            if hasattr(response, 'generations'):
                for gen_list in response.generations:
                    for generation in gen_list:
                        output_text += generation.text if hasattr(generation, 'text') else str(generation)
            else:
                output_text = str(response)
            output_tokens = len(output_text) // 2
            print(f"[DEBUG] Estimated tokens: {input_tokens}+{output_tokens}")

        # 获取响应文本用于记录
        response_text = ""
        if hasattr(response, 'generations') and response.generations:
            for gen_list in response.generations:
                for generation in gen_list:
                    response_text += generation.text if hasattr(generation, 'text') else str(generation)

        # 记录到tracker
        self.tracker.log(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=str(self._last_input)[:500] if self._last_input else None,
            response=str(response_text)[:500] if response_text else None,
        )

        print(f"[DEBUG] Logged to tracker: {input_tokens} input, {output_tokens} output")

        # 重置
        self._last_input = None


def estimate_tokens(text: str) -> int:
    """
    估算文本的token数量（粗略估算）

    对于中文：约1字符 = 1 token
    对于英文：约4字符 = 1 token

    Args:
        text: 要估算的文本

    Returns:
        估算的token数量
    """
    if not text:
        return 0

    # 统计中文字符
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fa5')
    # 统计非中文字符
    other_chars = len(text) - chinese_chars

    # 中文1字符≈1token，英文4字符≈1token
    return chinese_chars + other_chars // 4
