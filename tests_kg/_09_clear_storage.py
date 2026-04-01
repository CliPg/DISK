import os
import unittest.mock

from disk_kg.models import Entity, KnowledgeGraph, Relation, SQLiteConnector


def test_clear_storage():
    db_path = "test_clear_kg.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # 1. Prepare data and save to SQLite
    e1 = Entity(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])
    e2 = Entity(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0])
    r1 = Relation(start_entity=e1, end_entity=e2, label="knows", name="Alice knows Bob")

    kg = KnowledgeGraph()
    kg.add_entities([e1, e2])
    kg.add_relations([r1])

    print(f"Initial entities: {len(kg.entities)}")
    print(f"Initial relations: {len(kg.relations)}")

    connector = SQLiteConnector(db_path=db_path, graph_id="test_graph")
    kg.save_to(connector)

    # Verify data is saved
    kg_loaded = KnowledgeGraph()
    kg_loaded.load_from(connector)
    assert len(kg_loaded.entities) == 2
    assert len(kg_loaded.relations) == 1
    print("Data saved successfully.")

    # 2. Test clear with confirm="yes"
    print("Testing clear(confirm='yes')...")
    connector.clear(confirm="yes")

    kg_after_clear = KnowledgeGraph()
    kg_after_clear.load_from(connector)
    print(f"Entities after clear: {len(kg_after_clear.entities)}")
    print(f"Relations after clear: {len(kg_after_clear.relations)}")

    assert len(kg_after_clear.entities) == 0
    assert len(kg_after_clear.relations) == 0
    print("clear(confirm='yes') works!")

    # 3. Test clear with interactive 'y'
    # Repopulate data
    kg.save_to(connector)
    kg_test = KnowledgeGraph()
    kg_test.load_from(connector)
    assert len(kg_test.entities) == 2

    print("Testing interactive clear with 'y'...")
    with unittest.mock.patch("builtins.input", return_value="y"):
        connector.clear()

    kg_after_y = KnowledgeGraph()
    kg_after_y.load_from(connector)
    assert len(kg_after_y.entities) == 0
    print("Interactive clear with 'y' works!")

    # 4. Test clear with interactive 'n' (should not clear)
    kg.save_to(connector)
    kg_test = KnowledgeGraph()
    kg_test.load_from(connector)
    assert len(kg_test.entities) == 2

    print("Testing interactive clear with 'n'...")
    with unittest.mock.patch("builtins.input", return_value="n"):
        connector.clear()

    kg_after_n = KnowledgeGraph()
    kg_after_n.load_from(connector)
    assert len(kg_after_n.entities) == 2
    print("Interactive clear with 'n' correctly cancelled!")

    connector.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    print("All clear tests passed!")


if __name__ == "__main__":
    test_clear_storage()
