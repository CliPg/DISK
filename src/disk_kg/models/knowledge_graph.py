from disk_kg.models.distiller import TextBlock

from .connector import Connector


class Entity:
    def __init__(
        self,
        label: str,
        name: str,
        embedding=None,
        description: str = "",
        source_block: TextBlock = None,
    ):
        """
        Args:
            label (str): type of the entity
            name (str): entity name
            embedding: entity embedding vector
            description (str): detailed description of the entity in the text
            source_block (TextBlock): the text block where this entity was extracted from
        """
        self.label = label
        self.name = name
        self.embedding = embedding
        self.description = description
        self.source_block = source_block

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        data = {
            "label": self.label,
            "name": self.name,
            "description": self.description,
        }
        if self.source_block:
            data["source_block"] = self.source_block.to_dict()
        return data

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.label == other.label

    def __hash__(self):
        return hash((self.name, self.label))

    def __repr__(self):
        return f"Entity(label={self.label}, name={self.name})"


class Relation:
    def __init__(
        self,
        start_entity: Entity,
        end_entity: Entity,
        label: str,
        name: str,
        embedding=None,
        description: str = "",
        source_block: TextBlock = None,
    ):
        """
        Args:
            start_entity (Entity): the starting entity of the relation
            end_entity (Entity): the ending entity of the relation
            label (str): type of the relation
            name (str): relation name
            embedding: relation embedding vector
            description (str): detailed description of the relation in the text
            source_block (TextBlock): the text block where this relation was extracted from
        """
        self.start_entity = start_entity
        self.end_entity = end_entity
        self.label = label
        self.name = name
        self.embedding = embedding
        self.description = description
        self.source_block = source_block

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        data = {
            "start_entity": self.start_entity.to_dict(),
            "end_entity": self.end_entity.to_dict(),
            "label": self.label,
            "name": self.name,
            "description": self.description,
        }
        if self.source_block:
            data["source_block"] = self.source_block.to_dict()
        return data

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (
            self.start_entity.name == other.start_entity.name
            and self.end_entity.name == other.end_entity.name
            and self.label == other.label
            and self.name == other.name
        )

    def __hash__(self):
        return hash((self.start_entity.name, self.end_entity.name, self.label, self.name))

    def __repr__(self):
        return f"Relation({self.start_entity.name} -[{self.label}]-> {self.end_entity.name})"


class KnowledgeGraph:
    def __init__(self):
        self.entities: set[Entity] = set()
        self.relations: set[Relation] = set()

    def load_from(self, connector: Connector):
        """从存储器加载实体和关系"""
        self.entities = set(connector.get_all_entities())
        self.relations = set(connector.get_all_relations())

    def save_to(self, connector: Connector):
        """将当前的实体和关系保存到存储器"""
        connector.upsert_entities(list(self.entities))
        connector.upsert_relations(list(self.relations))

    def __add__(self, other):
        if not isinstance(other, KnowledgeGraph):
            return NotImplemented
        combined = KnowledgeGraph()
        combined.entities = self.entities.union(other.entities)
        combined.relations = self.relations.union(other.relations)
        return combined
