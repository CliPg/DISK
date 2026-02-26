#--------------------#
# extraction prompts #
#--------------------#

# Language detection
from .lang_detect import detect_document_language


def get_prompts(language=None):
    """
    Get prompts based on document language.

    Args:
        language: 'zh' for Chinese, 'en' for English, or None for auto-detection

    Returns:
        dict: Dictionary containing prompt strings for the specified language
    """
    if language == 'zh':
        return {
            'extract_entities': EXTRACT_ENTITIES_PROMPT_ZH,
            'extract_relations': EXTRACT_RELATIONS_PROMPT_ZH,
            'extract': EXTRACT_PROMPT_ZH,
            'extract_entities_and_relations': EXTRACT_ENTITIES_AND_RELATIONS_PROMPT_ZH,
        }
    else:
        return {
            'extract_entities': EXTRACT_ENTITIES_PROMPT,
            'extract_relations': EXTRACT_RELATIONS_PROMPT,
            'extract': EXTRACT_PROMPT,
            'extract_entities_and_relations': EXTRACT_ENTITIES_AND_RELATIONS_PROMPT,
        }


#--------------------#
# English prompts #
#--------------------#

# Prompt for extracting entities from text
EXTRACT_ENTITIES_PROMPT = """
You are an expert at extracting entities from text. Given the following text, identify and extract as many relevant entities as possible along with their labels, names, and descriptions. Be as specific as possible with the entities.
For each entity, provide a brief but informative description explaining what the entity represents in the context of the text.
You MUST respond with a valid JSON object only.
Do NOT include explanations, comments, text before or after the JSON.
If you add anything outside the JSON brackets, it will break the system.
Output only:
{
  "entities": [
    {
      "label": "Entity Label",
      "name": "Entity Name",
      "description": "A brief description of what this entity represents in the text"
    }
  ]
}
"""

# Prompt for extracting relations from text
EXTRACT_RELATIONS_PROMPT = """
You are an expert at extracting relations between entities from text. Given the following text, identify and extract all relevant relations along with their labels, names, and descriptions.
For each relation, provide a brief but informative description explaining the nature of this relationship in the context of the text.
You MUST respond with a valid JSON object only.
Do NOT include explanations, comments, text before or after the JSON.
If you add anything outside the JSON brackets, it will break the system.
Output only:
{
  "relations": [
    {
      "start_entity": "start_entity_name",
      "end_entity": "end_entity_name",
      "label": "Relation Label",
      "name": "Relation Name",
      "description": "A brief description of this relationship in the text"
    }
  ]
}
"""

# Prompt for extracting relations and entities from text
EXTRACT_PROMPT = """
You are an expert at extracting relations between entities from text. Given the following text, identify and extract as many relevant relations and entities as possible along with their labels, names, and descriptions. Be as specific as possible with the entities.
For each entity and relation, provide a brief but informative description explaining what it represents in the context of the text.
You MUST respond with a valid JSON object only.
Do NOT include explanations, comments, text before or after the JSON.
If you add anything outside the JSON brackets, it will break the system.
Output only:
{
  "relations": [
    {
      "start_entity":
        {
          "label": "Start Entity Label",
          "name": "Start Entity Name",
          "description": "Description of the start entity"
        },
      "end_entity":
        {
          "label": "End Entity Label",
          "name": "End Entity Name",
          "description": "Description of the end entity"
        },
      "label": "Relation Label",
      "name": "Relation Name",
      "description": "Description of the relation"
    }
  ]
}
"""

EXTRACT_ENTITIES_AND_RELATIONS_PROMPT = """
---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `label`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `name`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}

---Real Data to be Processed---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""


#--------------------#
# Chinese prompts #
#--------------------#

# 提取实体的中文提示词
EXTRACT_ENTITIES_PROMPT_ZH = """
你是一位从文本中提取实体的专家。给定以下文本，识别并提取尽可能多的相关实体，包括它们的标签、名称和描述。尽可能具体地提取实体。
对于每个实体，提供一个简洁但信息丰富的描述，解释该实体在文本上下文中代表什么。
你必须仅以有效的JSON对象格式响应。
不要包含解释、注释、JSON前后的文本。
如果在JSON括号外添加任何内容，会破坏系统。
仅输出以下格式：
{
  "entities": [
    {
      "label": "实体标签",
      "name": "实体名称",
      "description": "该实体在文本中所代表含义的简要描述"
    }
  ]
}
"""

# 提取关系的中文提示词
EXTRACT_RELATIONS_PROMPT_ZH = """
你是一位从文本中提取实体间关系的专家。给定以下文本，识别并提取所有相关的关系，包括它们的标签、名称和描述。
对于每个关系，提供一个简洁但信息丰富的描述，解释该关系在文本上下文中的性质。
你必须仅以有效的JSON对象格式响应。
不要包含解释、注释、JSON前后的文本。
如果在JSON括号外添加任何内容，会破坏系统。
仅输出以下格式：
{
  "relations": [
    {
      "start_entity": "起始实体名称",
      "end_entity": "结束实体名称",
      "label": "关系标签",
      "name": "关系名称",
      "description": "该关系在文本中的简要描述"
    }
  ]
}
"""

# 同时提取实体和关系的中文提示词
EXTRACT_PROMPT_ZH = """
你是一位从文本中提取实体间关系的专家。给定以下文本，识别并提取尽可能多的相关关系和实体，包括它们的标签、名称和描述。尽可能具体地提取实体。
对于每个实体和关系，提供一个简洁但信息丰富的描述，解释它在文本上下文中代表什么。
你必须仅以有效的JSON对象格式响应。
不要包含解释、注释、JSON前后的文本。
如果在JSON括号外添加任何内容，会破坏系统。
仅输出以下格式：
{
  "relations": [
    {
      "start_entity":
        {
          "label": "起始实体标签",
          "name": "起始实体名称",
          "description": "起始实体的描述"
        },
      "end_entity":
        {
          "label": "结束实体标签",
          "name": "结束实体名称",
          "description": "结束实体的描述"
        },
      "label": "关系标签",
      "name": "关系名称",
      "description": "关系的描述"
    }
  ]
}
"""

EXTRACT_ENTITIES_AND_RELATIONS_PROMPT_ZH = """
---角色---
你是一位知识图谱专家，负责从输入文本中提取实体和关系。

---指令---
1.  **实体提取与输出：**
    *   **识别：** 在输入文本中识别清晰定义且有意义的实体。
    *   **实体详情：** 对于每个识别出的实体，提取以下信息：
        *   `label`：实体的名称。如果实体名称不区分大小写，将每个重要单词的首字母大写（标题格式）。确保在整个提取过程中**命名一致**。
        *   `name`：使用以下类型之一对实体进行分类：`{entity_types}`。如果提供的实体类型都不适用，不要添加新的实体类型，将其归类为`其他`。
    *   **输出格式 - 实体：** 每个实体输出4个字段，用`{tuple_delimiter}`分隔，在一行上。第一个字段*必须*是字面字符串`entity`。
        *   格式：`entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **关系提取与输出：**
    *   **识别：** 识别先前提取的实体之间直接的、明确陈述的和有意义的关系。
    *   **N元关系分解：** 如果单个语句描述了涉及两个以上实体（N元关系）的关系，则将其分解为多个二元（双实体）关系对以进行单独描述。
        *   **示例：** 对于"Alice、Bob和Carol在项目X上合作"，提取二元关系，如"Alice与项目X合作"、"Bob与项目X合作"和"Carol与项目X合作"，或根据最合理的二元解释提取"Alice与Bob合作"。
    *   **关系详情：** 对于每个二元关系，提取以下字段：
        *   `source_entity`：源实体的名称。确保与实体提取**命名一致**。如果名称不区分大小写，将每个重要单词的首字母大写（标题格式）。
        *   `target_entity`：目标实体的名称。确保与实体提取**命名一致**。如果名称不区分大小写，将每个重要单词的首字母大写（标题格式）。
        *   `relationship_keywords`：一个或多个高级关键词，总结关系的总体性质、概念或主题。此字段中的多个关键字必须用逗号`,`分隔。**不要使用`{tuple_delimiter}`分隔此字段中的多个关键字。**
        *   `relationship_description`：对源实体和目标实体之间关系的性质的简要说明，为其联系提供清晰的依据。
    *   **输出格式 - 关系：** 每个关系输出5个字段，用`{tuple_delimiter}`分隔，在一行上。第一个字段*必须*是字面字符串`relation`。
        *   格式：`relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **分隔符使用协议：**
    *   `{tuple_delimiter}`是一个完整的原子标记，**不得填充内容**。它严格用作字段分隔符。
    *   **错误示例：** `entity{tuple_delimiter}东京<|location|>东京是日本的首都。`
    *   **正确示例：** `entity{tuple_delimiter}东京{tuple_delimiter}location{tuple_delimiter}东京是日本的首都。`

4.  **关系方向与重复：**
    *   除非明确说明，否则将所有关系视为**无向**。交换无向关系的源实体和目标实体不构成新关系。
    *   避免输出重复的关系。

5.  **输出顺序与优先级：**
    *   首先输出所有提取的实体，然后输出所有提取的关系。
    *   在关系列表中，优先输出与输入文本核心意义**最相关**的关系。

6.  **上下文与客观性：**
    *   确保所有实体名称和描述都以**第三人称**书写。
    *   明确命名主语或宾语；**避免使用代词**，如`本文`、`本文`、`我们公司`、`我`、`你`、`他/她`。

7.  **语言与专有名词：**
    *   整个输出（实体名称、关键字和描述）必须用`{language}`书写。
    *   专有名词（如人名、地名、组织名称）如果没有合适的、广泛接受的翻译，或会导致歧义，应保留其原始语言。

8.  **完成信号：** 仅在完全提取并输出所有实体和关系（遵循所有标准）后，输出字面字符串`{completion_delimiter}`。

---示例---
{examples}

---待处理的真实数据---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""