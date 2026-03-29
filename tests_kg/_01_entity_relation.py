# pass at 2026/3/22
from disk_kg import Entity, Relation

entities = [
    Entity(label="Person", name="Alice"),
    Entity(label="Person", name="Bob"),
    Entity(label="Company", name="Acme Corp"),
]

relations = [
    Relation(
        start_entity=entities[0],
        end_entity=entities[1],
        label="knows",
        name="Alice knows Bob",
    ),
    Relation(
        start_entity=entities[0],
        end_entity=entities[2],
        label="works_at",
        name="Alice works at Acme Corp",
    ),
    Relation(
        start_entity=entities[1],
        end_entity=entities[2],
        label="owns",
        name="Bob owns Acme Corp",
    ),
]
