from pydantic import BaseModel

"""The output structure LLM extracts information into"""

class EntitySchema(BaseModel):
    label: str
    name: str

class RelationSchema(BaseModel):
    start_entity: EntitySchema
    end_entity: EntitySchema
    label: str
    name: str

class EntitiesSchema(BaseModel):
    entities: list[EntitySchema]

class RelationsSchema(BaseModel):
    relations: list[RelationSchema]