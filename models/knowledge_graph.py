class Entity():
    def __init__(self, label:str, name:str, embedding):
        """
        Args:
            label (str): type of the entity
            name (str): entity name
            embedding: entity embedding vector
        """
        self.label = label
        self.name = name
        self.embedding = embedding


class Relationship():
    def __init__(self, start_entity:Entity, end_entity:Entity, label:str, name:str, embedding):
        """
        Args:
            start_entity (Entity): the starting entity of the relationship
            end_entity (Entity): the ending entity of the relationship
            label (str): type of the relationship
            name (str): relationship name
            embedding: relationship embedding vector
        """
        self.start_entity = start_entity
        self.end_entity = end_entity
        self.label = label
        self.name = name
        self.embedding = embedding


class KnowledgeGraph:
    def __init__(self):
        self.entities = []
        self.relationships = []

    def add_entity(self, entity:Entity):
        self.entities.append(entity)

    def add_relationship(self, relationship:Relationship):
        self.relationships.append(relationship)