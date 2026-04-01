**项目概述**

- **名称**: DISK
- **简介**: DISK（Domain Incremental conStruction of Knowledge graph）是一个从文档（当前以 PDF 为主）中蒸馏文本、抽取实体与关系、构建并合并领域知识图谱的工具集。项目通过将 PDF 文本块提取、调用 LLM 进行结构化信息抽取、对实体关系做向量化并合并近似实体，逐步构建知识图谱。

**设计目标**

- **可扩展的文档蒸馏**: 支持从 PDF 提取段落、表格和图片（含 OCR）的功能模块。
- **基于 LLM 的信息抽取**: 使用可替换的 LLM 与向量化接口完成实体与关系抽取以及嵌入生成。
- **语义合并与知识管理**: 提供实体/关系合并策略及知识图（KG）管理接口，便于增量构建与持久化。

**架构概览**

- **输入**: PDF 文档
- **阶段**: 蒸馏（distiller） -> 抽取（extractor） -> 合并（merger） -> 管理/构建（manager/models）
- **输出**: `KnowledgeGraph` 实例（包含 `Entity` 与 `Relation`）

**主要文件与模块说明**

- **`disk.py`**: 
  - **作用**: 提供 `DISK` 类作为主流程控制器，封装整个构建知识图的流水线。
  - **关键方法**: `build_knowledge_graph`（分步：蒸馏 -> 抽实体 -> 抽关系 -> 构建 KG），`build_knowledge_graph_single_extractor`（用单一抽取器同时抽实体与关系并合并）。

- **`distiller/pdf_distiller.py` (`PDFDistiller`)**:
  - **作用**: 使用 `PyMuPDF`（fitz）、`pdfplumber`、`PaddleOCR` 等从 PDF 中提取文本块、表格与图片+OCR。
  - **特点**: 将页面文本分块为「段落」，内置基本筛选（去除参考文献、短文本、非句子等）；提供表格与图片提取的多种策略与 TODO 优化点。

- **`extractor/`**:
  - **`extractor.py` (`Extractor`)**: 调用 `Parser`（基于 LangChain 风格）并使用预设 prompt（`utils.prompts.EXTRACT_PROMPT`）把文本解析为结构化 JSON（`RelationsSchema` / `EntitiesSchema`），随后为实体/关系生成 embedding 并返回嵌入化对象列表。
  - 项目里还有拆分的 `EntitiesExtractor` 与 `RelationsExtractor`（在导入点可见），用于分别提取实体或关系以支持分步流水线。

- **`merger/merger.py` (`Merger`)**:
  - **作用**: 基于实体 embedding 的余弦相似度（PyTorch 实现）合并相近实体，并在合并后更新/合并关系集合。
  - **阈值**: 可配置 `threshold`（默认 0.8），日志会输出相似度矩阵与合并信息到 `logs/` 目录。

- **`models/knowledge_graph.py`**:
  - **核心类**: `Entity`（`label`, `name`, `embedding`）、`Relation`（`start_entity`, `end_entity`, `label`, `name`, `embedding`）、`KnowledgeGraph`（保存实体/关系列表）。

- **`config/llm.py`**:
  - **作用**: 提供默认 LLM 与 embeddings 的配置（示例中使用 Qwen via `ChatTongyi` 与 `DashScopeEmbeddings`），并示范如何替换为 Ollama/本地模型。
  - **备注**: 真实使用前请在 `config/config.py`（或 `config/config.py` 中的 `api_key`）中配置 API key。

- **`utils/parser.py`**:
  - **作用**: 封装基于 LangChain 风格的 `JsonOutputParser` 与 `PromptTemplate`，将 LLM 的输出解析为 Pydantic 结构化对象。

- **其他工具**:
  - `utils/prompts.py`、`utils/schemas.py`: 定义 prompt 模板与 pydantic 输出 schema，用于约束 LLM 输出格式。

**运行与安装**

- **环境**: 推荐 Python 3.10
- **创建虚拟环境与安装依赖** (在 PowerShell 中运行):

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **配置 LLM API key**:
  - 编辑 `config/config.py` （或 `config/config.py` 中的 `api_key`）来填入你的 API Key。

- **示例：用 `DISK` 构建知识图** (Python 交互或脚本):

```python
from disk import DISK
from config.llm import llm, embeddings

disk = DISK(llm=llm, embeddings=embeddings)
kg = disk.build_knowledge_graph("path/to/your.pdf")
print(len(kg.entities), len(kg.relations))
```

**测试与示例**

- 仓库 `tests/` 下包含若干 Notebook 示例（`test_build_kg.ipynb`、`test_extractor.ipynb` 等），可作为快速上手的示例流程。

**日志与结果目录**

- 提取与合并过程会向 `../logs/` 写入诊断信息（相似度矩阵、提取段落、合并记录等）；提取输出也会追加到 `../results/`（`extracted_relations.json`/`extracted_entities.json`）。

**扩展点与注意事项**

- OCR、表格识别与多语言支持仍有优化空间（代码中标注了多处 TODO）。
- 当前实体合并基于 embedding 的余弦阈值，可能需要更复杂的语义/规则合并策略以减少误合并。
- LLM 调用和提示（`utils/prompts.py`）是抽取质量的关键，建议结合领域示例进一步工程化 prompt 与 schema。

**下一步建议**

- 添加单元测试覆盖关键模块（`distiller`、`extractor`、`merger`）以便回归检测。
- 提供一个简单的 CLI 或示例脚本，用于批量处理 PDF 并导出 KG（如 JSON/GraphML）。

---

文档已生成：`PROJECT_OVERVIEW.md`（位于仓库根目录）。如需我把这些更改提交到 git、或补充具体示例/图示（比如架构流程图），我可以继续操作。
