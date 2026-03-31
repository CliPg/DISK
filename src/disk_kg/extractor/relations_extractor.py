import json
import os

from disk_kg.models import RelationsSchema, Relation
from disk_kg.utils.lang_detect import detect_document_language
from disk_kg.utils.parser import Parser
from disk_kg.utils.prompts import EXTRACT_RELATIONS_PROMPT, get_prompts


class RelationsExtractor:
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
        self.results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
        )
        os.makedirs(self.results_dir, exist_ok=True)

    def extract_relations(self, text: str, pdf_path: str = None) -> list[Relation]:
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
            relations = self.parser.extract_information_as_json_from_text(
                text=text,
                output_structure=RelationsSchema,
                prompt=self.prompts["extract_relations"],
            )
        except Exception as e:
            print(f"Error during relation extraction: {e}")
            return None

        if not relations or "relations" not in relations or len(relations["relations"]) == 0:
            print("No relations found in the text.")
            return None

        embedded_relations = self.embed_relations(relations)

        with open(
            os.path.join(self.results_dir, "extracted_relations.json"), "a", encoding="utf-8"
        ) as f:
            json.dump(relations, f, ensure_ascii=False)
            f.write("\n")

        return embedded_relations

    def embed_relations(self, relations: RelationsSchema) -> list[Relation]:
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
                embedding=embedding,
                description=relation.get("description", ""),
            )
            embedded_relations.append(embedded_relation)

        return embedded_relations
