from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class Parser:
    def __init__(self, llm, embeddings):
        """
        Initialize the Parser with a language model and embeddings.

        Args:
            llm: The language model instance.
            embeddings: The embeddings instance.
        """
        self.llm = llm
        self.embeddings = embeddings

    def extract_information_as_json_from_text(self, text: str, output_structure, prompt: str) -> str:
        """
        Extract structured information in JSON format from the given text using the provided prompt.

        Args:
            text (str): The input text to be processed.
            output_structure: The pydantic model defining the output structure.
            prompt (str): The prompt guiding the extraction process.
        
        Returns:
            str: The extracted information in JSON format.
        """
        parser = JsonOutputParser(pydantic_object=output_structure)

        template = """
        Text: {text}
        
        Question: {prompt}
        Format_instructions: {format_instructions}
        Answer:
        """
        
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["text", "prompt", "format_instructions"]
        )
        
        chain = prompt_template | self.llm | parser
        
        answer = chain.invoke({
            "text": text,
            "prompt": prompt,
            "format_instructions": parser.get_format_instructions()
        })

        return answer