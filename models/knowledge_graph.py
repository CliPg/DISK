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

    def __eq__(self, other):
        """通过 name 和 label 判断实体是否相等"""
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.label == other.label

    def __hash__(self):
        """使 Entity 可哈希，用于集合去重"""
        return hash((self.name, self.label))

    def __repr__(self):
        return f"Entity(label={self.label}, name={self.name})"


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

    def __eq__(self, other):
        """通过起点、终点、类型和名称判断关系是否相等"""
        if not isinstance(other, Relation):
            return False
        return (self.start_entity.name == other.start_entity.name and
                self.end_entity.name == other.end_entity.name and
                self.label == other.label and
                self.name == other.name)

    def __hash__(self):
        """使 Relation 可哈希，用于集合去重"""
        return hash((self.start_entity.name, self.end_entity.name, self.label, self.name))

    def __repr__(self):
        return f"Relation({self.start_entity.name} -[{self.label}]-> {self.end_entity.name})"


class KnowledgeGraph:

    entities: list[Entity] = []
    relations: list[Relation] = []