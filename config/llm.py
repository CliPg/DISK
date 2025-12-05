from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from config.config import api_key

# ------------ Qwen API ------------ #
llm = ChatTongyi(
    api_key=api_key,
    model="qwen3-max",
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
