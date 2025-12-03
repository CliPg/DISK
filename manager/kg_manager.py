from models import KnowledgeGraph

class KGManager:

    def __init__(self, kg:KnowledgeGraph=None):
        self.kg = kg

    def add_entities(self, entities:list):
        for entity in entities:
            self.kg.entities.append(entity)

    def add_relations(self, relations:list):
        for relation in relations:
            self.kg.relations.append(relation)