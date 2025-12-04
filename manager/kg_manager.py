from models import KnowledgeGraph

class KGManager:

    def __init__(self, kg:KnowledgeGraph=None):
        self.kg = kg

    def add_entities(self, entities:list):
        for entity in entities:
            if entity in self.kg.entities:
                continue
            self.kg.entities.append(entity)

    def add_relations(self, relations:list):
        for relation in relations:
            if relation in self.kg.relations:
                continue
            self.kg.relations.append(relation)