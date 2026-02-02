import re

def bold_entities_in_text(text: str, entities: dict) -> str:
    """
    在文本中将命中的实体名称加粗（Markdown 格式）。
    """
    # 取出实体列表
    entity_names = [e["name"] for e in entities["entities"] if e.get("name")]

    # 按长度排序，可以避免短实体先匹配导致长实体无法匹配
    entity_names = sorted(set(entity_names), key=lambda x: len(x), reverse=True)

    # 避免重复替换：使用正则表达式保证只替换未被加粗的部分
    for name in entity_names:
        # \b 不适用于符号实体（如 GSub），所以使用更宽松的匹配
        pattern = rf"(?<!\*)({re.escape(name)})(?!\*)"
        text = re.sub(pattern, r"**\1**", text)

    return text