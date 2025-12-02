# DISK
Domain Incremental conStruction of Knowledge graph.

## Modules

### distill document

- pdf_distiller
    - extract **paragraphs** 
    - extract **tables**(to be improved)
    - extract **imgs**(to be improved)

- extract entities
- extract relationships
- construct knowledge graph

## Config

**env**
```
python3.10 -m venv .venv
pip install -r requirements.txt
```

**llm config**
```
cd config
touch config.py

# then configure your api key in this file
# current default llm is Qwen, 
# you can switch to other model in config/llm.py
```