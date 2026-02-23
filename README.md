# DISK
Domain Incremental conStruction of Knowledge graph.

## Overview

DISK is a comprehensive toolkit for extracting knowledge from PDF documents and building domain knowledge graphs through text distillation, entity/relation extraction, and semantic merging. The system provides a modular pipeline that transforms unstructured PDF documents into structured knowledge representations.

### Core Capabilities

- **Document Distillation**: Extract and validate text blocks, tables, and images from PDF documents
- **Entity Extraction**: Identify and extract domain entities with semantic embeddings
- **Relation Extraction**: Discover relationships between entities with contextual understanding
- **Knowledge Graph Construction**: Build and manage knowledge graphs with incremental updates
- **Semantic Merging**: Intelligently merge similar entities and relations using cosine similarity

## Architecture

### System Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        PDF[PDF Document]
    end

    subgraph "Distillation Layer"
        Distiller[PDF Distiller]
        TextBlocks[Validated Text Blocks]
    end

    subgraph "Extraction Layer"
        EntExtractor[Entity Extractor]
        RelExtractor[Relation Extractor]
        UnifiedExtractor[Unified Extractor]
        Entities[(Entities + Embeddings)]
        Relations[(Relations + Embeddings)]
    end

    subgraph "Processing Layer"
        Merger[Semantic Merger]
        Manager[KG Manager]
    end

    subgraph "Output Layer"
        KG[Knowledge Graph]
        Logs[Logs & Results]
    end

    subgraph "Configuration"
        Config[LLM Config]
        Embed[Embeddings Model]
    end

    PDF --> Distiller
    Distiller --> TextBlocks

    TextBlocks --> EntExtractor
    TextBlocks --> RelExtractor
    TextBlocks --> UnifiedExtractor

    EntExtractor --> Entities
    RelExtractor --> Relations
    UnifiedExtractor --> Entities
    UnifiedExtractor --> Relations

    Entities --> Merger
    Relations --> Merger
    Merger --> Manager

    Manager --> KG
    Manager --> Logs

    Config --> EntExtractor
    Config --> RelExtractor
    Config --> UnifiedExtractor
    Embed --> EntExtractor
    Embed --> RelExtractor
    Embed --> UnifiedExtractor
    Embed --> Merger

    style PDF fill:#e1f5fe
    style KG fill:#c8e6c9
    style Distiller fill:#fff3e0
    style Merger fill:#f3e5f5
    style Manager fill:#e8f5e9
```

### Module Structure

```mermaid
graph LR
    subgraph DISK
        DiskMain[disk.py<br/>Main Entry Point]

        subgraph Core
            Distiller[distiller/<br/>PDF Distillation]
            Extractor[extractor/<br/>Information Extraction]
            MergerMod[merger/<br/>Knowledge Merging]
            ManagerMod[manager/<br/>KG Management]
        end

        subgraph Support
            Models[models/<br/>Data Models]
            Utils[utils/<br/>Utilities]
            ConfigMod[config/<br/>Configuration]
        end
    end

    DiskMain --> Distiller
    DiskMain --> Extractor
    DiskMain --> MergerMod
    DiskMain --> ManagerMod

    Extractor --> Models
    MergerMod --> Models
    ManagerMod --> Models

    Distiller --> Utils
    Extractor --> Utils
    ManagerMod --> Utils

    DiskMain --> ConfigMod

    style DiskMain fill:#1976d2,color:#fff
    style Distiller fill:#ffa726
    style Extractor fill:#42a5f5
    style MergerMod fill:#ab47bc
    style ManagerMod fill:#66bb6a
```

### Data Flow

```mermaid
sequenceDiagram
    participant User
    participant DISK
    participant Distiller
    participant Extractor
    participant Merger
    participant Manager
    participant KG

    User->>DISK: build_knowledge_graph(pdf_path)
    DISK->>Distiller: extract_text_blocks(pdf)
    Distiller-->>DISK: validated_text_blocks

    loop For each text block
        DISK->>Extractor: extract_entities(text)
        Extractor-->>DISK: entities + embeddings

        DISK->>Extractor: extract_relations(text)
        Extractor-->>DISK: relations + embeddings

        DISK->>Merger: merge(new, existing)
        Merger-->>DISK: merged entities/relations
    end

    DISK->>Manager: add_entities(entities)
    DISK->>Manager: add_relations(relations)
    Manager->>KG: update_knowledge_graph
    DISK-->>User: Knowledge Graph
```

## Modules

### Distillation Module (distiller/)

- **pdf_distiller**
    - extract **paragraphs** with intelligent validation
    - extract **tables**(to be improved)
    - extract **imgs**(to be improved)
    - filter out low-quality text blocks (references, incomplete sentences)

### Extraction Module (extractor/)

- **entities_extractor**
    - extract domain entities with labels and descriptions
    - generate semantic embeddings for each entity

- **relations_extractor**
    - extract relationships between entities
    - generate semantic embeddings for each relation

- **extractor** (unified)
    - extract both entities and relations in a single pass
    - optimized for incremental processing

### Processing Modules

- **extract entities**
- **extract relationships**
- **semantic merging** (merger/)
    - merge similar entities using cosine similarity
    - update relations after entity merging
    - configurable threshold (default: 0.8)
- **construct knowledge graph** (manager/)
    - incremental knowledge graph construction
    - deduplication of entities and relations

## Config

**env**
```bash
# use uv to manage the environment
uv venv
uv sync
```

**LLM Configuration**

1. Copy the example configuration file:
```bash
cp config.example.toml config.toml
```

2. Edit `config.toml` to set your API keys and preferences:

```toml
[disk]
llm = "openai"  # Choose provider: openai, qwen, ollama, etc.

[disk.embeddings]
model = "text-embedding-3-small"
api_key = "ai-..."
api_url = "https://api.openai.com/v1"

[model.openai]
api_url = "https://api.openai.com/v1"
api_key = "ai-..."
model = "gpt-4o"

[model.other]
api_url = "https://api.otherprovider.com/v1"
api_key = "sk-..."
model = "gpt-4o"
```

3. Supported providers:
   - **OpenAI** (default)
   - **Qwen** (DashScope)
   - **Kimi** (Moonshot)
   - **Ollama** (Local)
   - **All other providers** that support OpenAI-compatible APIs

You can switch providers by changing the `llm` field in `[disk]` or using the runtime `switch()` function.


## Contrast

### merge
- itext2kg
```
[INFO] Wohoo! Entity was matched --- [poor deep semantic understanding in traditional ie models:Limitation] --merged--> [cosine similarity ignores deep semantic differences:Limitation]
```