from utils.parser import Parser
from utils.schemas import Relations
from utils.prompts import EXTRACT_RELATIONS_PROMPT

class RelationsExtractor:
    
    def __init__(self, llm, embeddings):
        self.parser = Parser(llm=llm, embeddings=embeddings)

    def extract_entities(self, text:str):
        relations = self.parser.extract_information_as_json_from_text(
            text=text,
            output_structure=Relations,
            prompt=EXTRACT_RELATIONS_PROMPT
        )

        with open("relations_output.json", "w") as f:
            f.write(str(relations))
        return relations
