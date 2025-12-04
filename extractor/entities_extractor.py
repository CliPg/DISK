from utils.parser import Parser
from utils.schemas import EntitiesSchema
from utils.prompts import EXTRACT_ENTITIES_PROMPT
from models import Entity, Relation, KnowledgeGraph
import json

class EntitiesExtractor:
    
    def __init__(self, llm, embeddings):
        self.parser = Parser(llm=llm, embeddings=embeddings)

    def extract_entities(self, text:str) -> list[Entity] | None:
        """
        Extract entities from the given text.

        Args:
            text (str): The input text from which to extract entities.

        Returns:
            Entities: The extracted entities structured as per the Entities schema.
        """
        try:
            entities = self.parser.extract_information_as_json_from_text(
                text=text,
                output_structure=EntitiesSchema,
                prompt=EXTRACT_ENTITIES_PROMPT
            )
        except Exception as e:
            print(f"Error during entity extraction: {e}")
            return None

        if len(entities["entities"]) == 0:
            print("No entities found in the text.")
            return None

        embedded_entities = self.embed_entities(entities)


        with open("../results/extracted_entities.json", "a", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False)
            f.write("\n")

        return embedded_entities
    
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
            embedding = self.parser.embeddings.embed_query(entity["name"])
            embedded_entity = Entity(
                label=entity["label"],
                name=entity["name"],
                embedding=embedding
            )
            embedded_entities.append(embedded_entity)

        return embedded_entities
    
    