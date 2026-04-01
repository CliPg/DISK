from pydantic import BaseModel, Field

from .connector import Connector
from .knowledge_graph import Entity, KnowledgeGraph, Relation
from .merger import Merger
from .neo4j_connector import Neo4jConnector
from .sqlite_connector import SQLiteConnector


class EntitySchema(BaseModel):
    label: str = Field(description="The label or type of the entity")
    name: str = Field(description="The name of the entity")
    description: str = Field(default="", description="A brief description of the entity")


class RelationSchema(BaseModel):
    start_entity: EntitySchema = Field(description="The start entity of the relation")
    end_entity: EntitySchema = Field(description="The end entity of the relation")
    label: str = Field(description="The label or type of the relation")
    name: str = Field(description="The name of the relation")
    description: str = Field(default="", description="A brief description of the relation")


class RelationsSchema(BaseModel):
    relations: list[RelationSchema] = Field(description="List of extracted relations")


class EntitiesSchema(BaseModel):
    entities: list[EntitySchema] = Field(description="List of extracted entities")


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
