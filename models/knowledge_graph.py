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


class Relation():
    def __init__(self, start_entity:Entity, end_entity:Entity, label:str, name:str, embedding):
        """
        Args:
            start_entity (Entity): the starting entity of the relation
            end_entity (Entity): the ending entity of the relation
            label (str): type of the relation
            name (str): relation name
            embedding: relation embedding vector
        """
        self.start_entity = start_entity
        self.end_entity = end_entity
        self.label = label
        self.name = name
        self.embedding = embedding


class KnowledgeGraph:

    def __init__(self, entities:list[Entity]=None, relations:list[Relation]=None):
        self.entities = entities
        self.relations = relations

    def add_entity(self, entity:Entity):
        self.entities.append(entity)

    def add_relation(self, relation:Relation):
        self.relations.append(relation)