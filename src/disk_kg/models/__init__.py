from .connector import Connector
from .knowledge_graph import Entity, KnowledgeGraph, Relation
from .merger import Merger
from .neo4j_connector import Neo4jConnector
from .sqlite_connector import SQLiteConnector

EntitySchema = Entity
EntitiesSchema = list[EntitySchema]
RelationSchema = Relation
RelationsSchema = list[RelationSchema]

__all__ = [
    "Connector",
    "Neo4jConnector",
    "SQLiteConnector",
    "Entity",
    "Relation",
    "KnowledgeGraph",
    "Merger",
    "EntitySchema",
    "EntitiesSchema",
    "RelationSchema",
    "RelationsSchema",
]
