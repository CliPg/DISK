from pydantic import BaseModel

"""The output structure LLM extracts information into"""

class Entitiy(BaseModel):
    label: str
    name: str

class Relation(BaseModel):
    start_entity: Entitiy
    end_entity: Entitiy
    label: str
    name: str

class Entities(BaseModel):
    entities: list[Entitiy]

class Relations(BaseModel):
    relations: list[Relation]