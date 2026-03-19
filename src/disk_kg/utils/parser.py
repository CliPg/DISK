from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class Parser:
    def __init__(self, llm, embeddings, token_callback=None):
        """
        Initialize the Parser with a language model and embeddings.

        Args:
            llm: The language model instance.
            embeddings: The embeddings instance.
            token_callback: Optional token tracking callback handler.
        """
        self.llm = llm
        self.embeddings = embeddings
        self.token_callback = token_callback

    def extract_information_as_json_from_text(
        self, text: str, output_structure, prompt: str
    ) -> str:
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
            template=template, input_variables=["text", "prompt", "format_instructions"]
        )

        chain = prompt_template | self.llm | parser

        # 准备调用参数
        invoke_args = {
            "text": text,
            "prompt": prompt,
            "format_instructions": parser.get_format_instructions(),
        }

        # 如果有回调，通过config传递
        if self.token_callback:
            from langchain_core.runnables import RunnableConfig

            config = RunnableConfig(callbacks=[self.token_callback])
            print(f"[DEBUG] Invoking chain with token_callback: {self.token_callback}")
            extracted_information = chain.invoke(invoke_args, config=config)
        else:
            print(f"[DEBUG] Invoking chain without token_callback")
            extracted_information = chain.invoke(invoke_args)

        return extracted_information
