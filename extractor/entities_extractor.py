from utils.parser import Parser
from utils.schemas import Entities
from utils.prompts import EXTRACT_ENTITIES_PROMPT

class EntitiesExtractor:
    
    def __init__(self, llm, embeddings):
        self.parser = Parser(llm=llm, embeddings=embeddings)

    def extract_entities(self, text:str):
        entities = self.parser.extract_information_as_json_from_text(
            text=text,
            output_structure=Entities,
            prompt=EXTRACT_ENTITIES_PROMPT
        )

        with open("entities_output.json", "w") as f:
            f.write(str(entities))
        return entities