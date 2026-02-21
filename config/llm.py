import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 从 .env 文件加载环境变量
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("LLM_API_KEY")
model = os.getenv("LLM_MODEL", "qwen-plus")

# ------------ Qwen API ------------ #
llm = ChatTongyi(
    api_key=api_key,
    model=model,
    temperature=0,
    max_retries=3
)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)


# -------- Ollama Llama3 ----------- #
# llm = ChatOllama(
#     model="llama3",
#     temperature=0,
# )
# 
# embeddings = OllamaEmbeddings(
#     model="llama3",
# )
