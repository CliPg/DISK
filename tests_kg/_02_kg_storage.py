import os

from disk_kg.models import Entity, KnowledgeGraph, Relation, SQLiteConnector


def test_kg_merge_and_storage():
    # 1. Prepare data
    e1 = Entity(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])
    e2 = Entity(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0])
    r1 = Relation(start_entity=e1, end_entity=e2, label="knows", name="Alice knows Bob")

    kg1 = KnowledgeGraph()
    kg1.add_entities([e1, e2])
    kg1.add_relations([r1])

    # Same Alice, but slightly different embedding (should merge if threshold is 0.8)
    e3 = Entity(label="Person", name="Alice", embedding=[0.9, 0.1, 0.0])
    e4 = Entity(label="Company", name="Acme", embedding=[0.0, 0.0, 1.0])
    r2 = Relation(start_entity=e3, end_entity=e4, label="works_at", name="Alice works at Acme")

    kg2 = KnowledgeGraph()
    kg2.add_entities([e3, e4])
    kg2.add_relations([r2])

    # 2. Test Merge (+)
    print("Testing KG merge...")
    kg_combined = kg1 + kg2
    print(f"Combined entities: {len(kg_combined.entities)}")
    print(f"Combined relations: {len(kg_combined.relations)}")

    # Alice should be merged because they have high similarity
    # e1 and e3 similarity is high.
    # Let's check.

    # 3. Test Storage (SQLite)
    db_path = "test_kg.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"Testing SQLite storage at {db_path}...")
    connector = SQLiteConnector(db_path=db_path)
    kg_combined.save_to(connector)

    # Load back
    kg_loaded = KnowledgeGraph()
    kg_loaded.load_from(connector)
    print(f"Loaded entities: {len(kg_loaded.entities)}")
    print(f"Loaded relations: {len(kg_loaded.relations)}")

    connector.close()

    assert len(kg_loaded.entities) == len(kg_combined.entities)
    assert len(kg_loaded.relations) == len(kg_combined.relations)
    print("Test passed!")


if __name__ == "__main__":
    test_kg_merge_and_storage()
