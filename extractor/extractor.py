from utils.parser import Parser
from utils.schemas import RelationsSchema, EntitiesSchema, EntitySchema
from utils.prompts import EXTRACT_PROMPT
from models import Relation, Entity
from typing import List, Tuple
import json
import numpy as np
import os

class Extractor:

    def __init__(self, llm, embeddings,  entity_label_weight:float=0.0, entity_name_weight:float=1.0):
        self.parser = Parser(llm=llm, embeddings=embeddings)
        self.entity_label_weight = entity_label_weight
        self.entity_name_weight = entity_name_weight
        self.entity_cache = {} # save entity embeddings
        # Set results directory to project root/results
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(self.results_dir, exist_ok=True)

    def extract_relations_and_entities(self, text:str) -> Tuple[List[Relation], List[Entity]] | None:
        """
        Extract relations from the given text.
        
        Args:
            text (str): The input text from which to extract relations.

        Returns:
            Relations: The extracted relations structured as per the Relations schema.
        """
        try:
            print("Calling llm...")
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

        print("Embedding relations and entities...")
        embedded_relations = self.embed_relations(relations)
        embedded_entities = self.embed_entities(entities)

        with open(os.path.join(self.results_dir, "extracted_relations.json"), "a", encoding="utf-8") as f:
            json.dump(relations, f, ensure_ascii=False)
            f.write("\n")

        with open(os.path.join(self.results_dir, "extracted_entities.json"), "a", encoding="utf-8") as f:
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
            relation_name = relation.get("name")
            relation_label = relation.get("label")
            start_entity = relation.get("start_entity")
            end_entity = relation.get("end_entity")

            if not self.is_valid_string(relation_name) or not self.is_valid_string(relation_label):
                print("invalid relation name or label, skipping...")
                continue

            embedding = self.parser.embeddings.embed_query(relation["name"])
            embedded_relation = Relation(
                start_entity=self.embed_entity(relation["start_entity"]),
                end_entity=self.embed_entity(relation["end_entity"]),
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
            name = entity.get("name")
            label = entity.get("label")
            if not self.is_valid_string(name) or not self.is_valid_string(label):
                print("invalid entity name or label, skipping...")
                continue

            embedded_entity = self.embed_entity(entity)
            embedded_entities.append(embedded_entity)

        return embedded_entities
    
    def embed_entity(self, entity:EntitySchema) -> Entity:
        """
        Generate embedding for a single entity.

        Args:
            entity (Entity): The entity to be embedded.
        
        Returns:
            Entity: The entity with its embedding.
        """
        name = entity.get("name")
        label = entity.get("label")

        key = (entity["name"], entity["label"])
        if key in self.entity_cache:
            return self.entity_cache[key]

        name_embedding = self.parser.embeddings.embed_query(entity["name"])
        label_embedding = self.parser.embeddings.embed_query(entity["label"])

        embedding = self.entity_name_weight * np.array(name_embedding) + self.entity_label_weight * np.array(label_embedding)
        embedding = embedding.tolist()

        embedded_entity = Entity(
            label=entity["label"],
            name=entity["name"],
            embedding=embedding
        )

        self.entity_cache[key] = embedded_entity

        return embedded_entity

    def is_valid_string(self, value):
        if not isinstance(value, str):
            return False
        if not value.strip():
            return False
        return True
