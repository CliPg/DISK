from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from config.config import api_key

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

