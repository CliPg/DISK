from utils.parser import Parser
from utils.schemas import RelationsSchema, EntitiesSchema
from utils.prompts import EXTRACT_PROMPT
from models import Relation, Entity
from typing import List, Tuple
import json
import numpy as np

class Extractor:
    
    def __init__(self, llm, embeddings,  entity_label_weight:float=0.0, entity_name_weight:float=1.0):
        self.parser = Parser(llm=llm, embeddings=embeddings)
        self.entity_label_weight = entity_label_weight
        self.entity_name_weight = entity_name_weight

    def extract_relations_and_entities(self, text:str) -> Tuple[List[Relation], List[Entity]] | None:
        """
        Extract relations from the given text.
        
        Args:
            text (str): The input text from which to extract relations.

        Returns:
            Relations: The extracted relations structured as per the Relations schema.
        """
        try:
            relations = self.parser.extract_information_as_json_from_text(
                text=text,
                output_structure=RelationsSchema,
                prompt=EXTRACT_PROMPT
            )
        except Exception as e:
            print(f"Error during relation extraction: {e}")
            return None

        if not relations or "relations" not in relations or len(relations["relations"]) == 0:
            print("No relations found in the text.")
            return None

        entities = self.extract_entities(relations)

        embedded_relations = self.embed_relations(relations)
        embedded_entities = self.embed_entities(entities)

        with open("../results/extracted_relations.json", "a", encoding="utf-8") as f:
            json.dump(relations, f, ensure_ascii=False)
            f.write("\n")

        with open("../results/extracted_entities.json", "a", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False)
            f.write("\n")

        return embedded_relations, embedded_entities

    def extract_entities(self, relations:RelationsSchema) -> list[Entity]:
        """
        Extract entities from the given relations.

        Args:
            relations (Relations): The extracted relations.
        
        Returns:
            Entities: The extracted entities structured as per the Entities schema.
        """
        entities = []

        for relation in relations["relations"]:
            start_entity = relation["start_entity"]
            end_entity = relation["end_entity"]
            if start_entity not in entities:
                entities.append(start_entity)
            if end_entity not in entities:
                entities.append(end_entity)

        return {"entities": entities}
            

    def embed_relations(self, relations:RelationsSchema) -> list[Relation]:
        """
        Generate embeddings for the extracted relations.

        Args:
            relations (Relations): The extracted relations.

        Returns:
            Relations: The relations with their embeddings.
        """
        embedded_relations = []

        for relation in relations["relations"]:
            if relation["name"] == None or relation["label"] == None:
                continue
            embedding = self.parser.embeddings.embed_query(relation["name"])
            embedded_relation = Relation(
                start_entity=relation["start_entity"],
                end_entity=relation["end_entity"],
                label=relation["label"],
                name=relation["name"],
                embedding=embedding
            )
            embedded_relations.append(embedded_relation)

        return embedded_relations
    
    def embed_entities(self, entities:EntitiesSchema) -> list[Entity]:
        """
        Generate embeddings for the extracted entities.

        Args:
            entities (Entities): The extracted entities.

        Returns:
            Entities: The entities with their embeddings.

        """
        embedded_entities = []

        for entity in entities["entities"]:
            if entity["name"] == None or entity["label"] == None:
                continue
            name_embedding = self.parser.embeddings.embed_query(entity["name"])
            label_embedding = self.parser.embeddings.embed_query(entity["label"])

            embedding = self.entity_name_weight * np.array(name_embedding) + self.entity_label_weight * np.array(label_embedding)
            embedding = embedding.tolist()

            embedded_entity = Entity(
                label=entity["label"],
                name=entity["name"],
                embedding=embedding
            )
            embedded_entities.append(embedded_entity)

        return embedded_entities
