from utils.parser import Parser
from utils.schemas import RelationsSchema
from utils.prompts import EXTRACT_RELATIONS_PROMPT
from models import Relation
import tqdm

class RelationsExtractor:
    
    def __init__(self, llm, embeddings):
        self.parser = Parser(llm=llm, embeddings=embeddings)

    def extract_relations(self, text:str) -> list[Relation]:
        relations = self.parser.extract_information_as_json_from_text(
            text=text,
            output_structure=RelationsSchema,
            prompt=EXTRACT_RELATIONS_PROMPT
        )

        if len(relations["relations"]) == 0:
            print("No relations found in the text.")
            return None

        embedded_relations = self.embed_relations(relations)

        with open("relations_output.json", "w") as f:
            f.write(str(relations))

        return embedded_relations

    def embed_relations(self, relations:RelationsSchema) -> list[Relation]:
        """
        Generate embeddings for the extracted relations.

        Args:
            relations (Relations): The extracted relations.

        Returns:
            Relations: The relations with their embeddings.
        """
        embedded_relations = []

        for relation in tqdm(relations["relations"]):
            embedding = self.parser.embeddings.embed_query(relation["name"])
            embedded_relation = Relation(
                entity1=relation["start_entity"],
                entity2=relation["end_entity"],
                label=relation["label"],
                name=relation["name"],
                embedding=embedding
            )
            embedded_relations.append(embedded_relation)

        return embedded_relations
