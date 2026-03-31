import json
import os

import numpy as np

from disk_kg.models import (
    EntitiesSchema,
    Entity,
    EntitySchema,
    Relation,
    RelationsSchema,
)
from disk_kg.utils.lang_detect import detect_document_language
from disk_kg.utils.parser import Parser
from disk_kg.utils.prompts import EXTRACT_PROMPT, get_prompts


class Extractor:
    def __init__(
        self,
        llm,
        embeddings,
        entity_label_weight: float = 0.0,
        entity_name_weight: float = 0.3,
        entity_description_weight: float = 0.7,
        language: str = None,
        token_callback=None,
    ):
        """
        Args:
            llm: Language model instance
            embeddings: Embeddings instance
            entity_label_weight: Weight for entity label in embedding
            entity_name_weight: Weight for entity name in embedding
            entity_description_weight: Weight for entity description in embedding
            language: 'zh' for Chinese, 'en' for English, or None for auto-detection
            token_callback: Token tracking callback handler
        """
        self.parser = Parser(llm=llm, embeddings=embeddings, token_callback=token_callback)
        self.entity_label_weight = entity_label_weight
        self.entity_name_weight = entity_name_weight
        self.entity_description_weight = entity_description_weight
        self.language = language
        self.prompts = get_prompts(language)
        self.entity_cache = {}  # save entity embeddings
        self.token_callback = token_callback
        # Set results directory to project root/results
        self.results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
        )
        os.makedirs(self.results_dir, exist_ok=True)

    def extract_relations_and_entities(
        self, text: str, pdf_path: str = None
    ) -> tuple[list[Relation], list[Entity]] | None:
        """
        Extract relations from the given text.

        Args:
            text (str): The input text from which to extract relations.
            pdf_path (str): Optional path to PDF file for language detection.

        Returns:
            Relations: The extracted relations structured as per the Relations schema.
        """
        # Auto-detect language if not set and pdf_path is provided
        if self.language is None and pdf_path:
            detected_lang = detect_document_language(file_path=pdf_path, text_content=text[:500])
            self.prompts = get_prompts(detected_lang)

        try:
            print("Calling llm...")
            relation_str = self.parser.extract_information_as_json_from_text(
                text=text, output_structure=RelationsSchema, prompt=self.prompts["extract"]
            )
        except Exception as e:
            print(f"Error during relation extraction: {e}")
            return None

        if not relations or "relations" not in relations or len(relations) == 0:
            print("No relations found in the text.")
            return None

        entities = self.extract_entities(relations)

        print("Embedding relations and entities...")
        embedded_relations = self.embed_relations(relations)
        embedded_entities = self.embed_entities(entities)

        with open(
            os.path.join(self.results_dir, "extracted_relations.json"), "a", encoding="utf-8"
        ) as f:
            json.dump(relations, f, ensure_ascii=False)
            f.write("\n")

        with open(
            os.path.join(self.results_dir, "extracted_entities.json"), "a", encoding="utf-8"
        ) as f:
            json.dump(entities, f, ensure_ascii=False)
            f.write("\n")

        return embedded_relations, embedded_entities

    def extract_entities(self, relations: RelationsSchema) -> EntitiesSchema:
        """
        Extract entities from the given relations.

        Args:
            relations (Relations): The extracted relations.

        Returns:
            Entities: The extracted entities structured as per the Entities schema.
        """
        entities = []

        for relation in relations:
            start_entity = relation.start_entity
            end_entity = relation.end_entity
            if start_entity not in entities:
                entities.append(start_entity)
            if end_entity not in entities:
                entities.append(end_entity)

        return entities

    def embed_relations(self, relations: RelationsSchema) -> RelationsSchema:
        """
        Generate embeddings for the extracted relations.

        Args:
            relations (Relations): The extracted relations.

        Returns:
            Relations: The relations with their embeddings.
        """
        embedded_relations = []

        for relation in relations:
            relation_name = relation.name
            relation_label = relation.label

            if not self.is_valid_string(relation_name) or not self.is_valid_string(relation_label):
                print("invalid relation name or label, skipping...")
                continue

            embedding = self.parser.embeddings.embed_query(relation.name)
            embedded_relation = Relation(
                start_entity=self.embed_entity(relation.start_entity),
                end_entity=self.embed_entity(relation.end_entity),
                label=relation.label,
                name=relation.name,
                embedding=embedding,
                description=relation.description,
            )
            embedded_relations.append(embedded_relation)

        return embedded_relations

    def embed_entities(self, entities: EntitiesSchema) -> list[Entity]:
        """
        Generate embeddings for the extracted entities.

        Args:
            entities (Entities): The extracted entities.

        Returns:
            Entities: The entities with their embeddings.

        """
        embedded_entities = []

        for entity in entities:
            name = entity.name
            label = entity.label
            if not self.is_valid_string(name) or not self.is_valid_string(label):
                print("invalid entity name or label, skipping...")
                continue

            embedded_entity = self.embed_entity(entity)
            embedded_entities.append(embedded_entity)

        return embedded_entities

    def embed_entity(self, entity: EntitySchema) -> Entity:
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
        description_embedding = self.parser.embeddings.embed_query(entity.get("description", ""))

        embedding = (
            self.entity_name_weight * np.array(name_embedding)
            + self.entity_label_weight * np.array(label_embedding)
            + self.entity_description_weight * np.array(description_embedding)
        )
        embedding = embedding.tolist()

        embedded_entity = Entity(
            label=entity["label"],
            name=entity["name"],
            embedding=embedding,
            description=entity.get("description", ""),
        )

        self.entity_cache[key] = embedded_entity

        return embedded_entity

    def is_valid_string(self, value):
        if not isinstance(value, str):
            return False
        if not value.strip():
            return False
        return True
