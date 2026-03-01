from utils.parser import Parser
from utils.schemas import EntitiesSchema
from utils.prompts import EXTRACT_ENTITIES_PROMPT
from utils.prompts import get_prompts
from utils.lang_detect import detect_document_language
from models import Entity, Relation, KnowledgeGraph
import json
import os

class EntitiesExtractor:

    def __init__(self, llm, embeddings, language: str = None, token_callback=None):
        """
        Args:
            llm: Language model instance
            embeddings: Embeddings instance
            language: 'zh' for Chinese, 'en' for English, or None for auto-detection
            token_callback: Token tracking callback handler
        """
        self.parser = Parser(llm=llm, embeddings=embeddings, token_callback=token_callback)
        self.language = language
        self.prompts = get_prompts(language)
        # Set results directory to project root/results
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(self.results_dir, exist_ok=True)

    def extract_entities(self, text:str, pdf_path: str = None) -> list[Entity] | None:
        """
        Extract entities from the given text.

        Args:
            text (str): The input text from which to extract entities.
            pdf_path (str): Optional path to PDF file for language detection.

        Returns:
            Entities: The extracted entities structured as per the Entities schema.
        """
        # Auto-detect language if not set and pdf_path is provided
        if self.language is None and pdf_path:
            detected_lang = detect_document_language(file_path=pdf_path, text_content=text[:500])
            self.prompts = get_prompts(detected_lang)

        try:
            entities = self.parser.extract_information_as_json_from_text(
                text=text,
                output_structure=EntitiesSchema,
                prompt=self.prompts['extract_entities']
            )
        except Exception as e:
            print(f"Error during entity extraction: {e}")
            return None

        if not entities or "entities" not in entities or len(entities["entities"]) == 0:
            print("No entities found in the text.")
            return None
                
        embedded_entities = self.embed_entities(entities)

        with open(os.path.join(self.results_dir, "extracted_entities.json"), "a", encoding="utf-8") as f:
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
            if entity["name"] == None or entity["label"] == None:
                continue
            embedding = self.parser.embeddings.embed_query(entity["name"])
            embedded_entity = Entity(
                label=entity["label"],
                name=entity["name"],
                embedding=embedding,
                description=entity.get("description", "")
            )
            embedded_entities.append(embedded_entity)

        return embedded_entities
    
    