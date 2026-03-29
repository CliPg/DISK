# pass at 2026/3/29
from disk_kg import KnowledgeGraph

from ._01_entity_relation import entities, relations

kg = KnowledgeGraph()
kg.add_entities(entities)
kg.add_relations(relations)
