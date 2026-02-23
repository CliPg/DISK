from config.llm import embeddings as embedder

text_to_embedding = ["风急天高猿啸哀", "渚清沙白鸟飞回", "无边落木萧萧下", "不尽长江滚滚来"]
# 调用 Embedding 模型
for index, text in enumerate(text_to_embedding):
    print(f"Input: {text}, length of input: {len(text)}")
    result_embeddings = embedder.embed_query(text)
    # result_embeddings 是一个 float list
    print(f"Dimension of embeddings: {len(result_embeddings)}")
    print(f"Input: {text}, embedding snippet: {result_embeddings[:5]}...")
