import sqlite3
from typing import Any

from disk_kg.distiller import TextBlock

from .connector import Connector
from .knowledge_graph import Entity, Relation


class SQLiteConnector(Connector):
    """
    A class to connect to a SQLite database and manage knowledge graph data.
    """

    def __init__(self, db_path: str = "knowledge_graph.db", graph_id: str | None = None):
        super().__init__(graph_id)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        # Create TextBlocks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS text_blocks (
                block_id INTEGER,
                page_number INTEGER,
                graph_id TEXT,
                text TEXT,
                file_path TEXT,
                PRIMARY KEY (block_id, page_number, graph_id)
            )
        """
        )

        # Create Entities table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT,
                label TEXT,
                graph_id TEXT,
                description TEXT,
                embedding BLOB,
                source_block_id INTEGER,
                source_page INTEGER,
                PRIMARY KEY (name, label, graph_id)
            )
        """
        )

        # Create Relations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                start_name TEXT,
                start_label TEXT,
                end_name TEXT,
                end_label TEXT,
                label TEXT,
                name TEXT,
                graph_id TEXT,
                description TEXT,
                embedding BLOB,
                source_block_id INTEGER,
                source_page INTEGER,
                PRIMARY KEY (start_name, end_name, label, graph_id)
            )
        """
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

    def run_query(self, query: str, parameters: dict | None = None) -> Any:
        cursor = self.conn.cursor()
        if parameters:
            cursor.execute(query, parameters)
        else:
            cursor.execute(query)
        self.conn.commit()
        return cursor.fetchall()

    def _upsert_text_block(self, block: TextBlock):
        query = """
        INSERT INTO text_blocks (block_id, page_number, graph_id, text, file_path)
        VALUES (:block_id, :page_number, :graph_id, :text, :file_path)
        ON CONFLICT(block_id, page_number, graph_id) DO UPDATE SET
            text = excluded.text,
            file_path = excluded.file_path
        """
        params = {
            "block_id": block.block_id,
            "page_number": block.page_number,
            "graph_id": self.graph_id or "default",
            "text": block.text,
        }
        self.run_query(query, params)

    def upsert_entities(self, entities: list[Entity]):
        for entity in entities:
            if entity.source_block:
                self._upsert_text_block(entity.source_block)

            query = """
            INSERT INTO entities (name, label, graph_id, description, embedding, source_block_id, source_page)
            VALUES (:name, :label, :graph_id, :description, :embedding, :source_block_id, :source_page)
            ON CONFLICT(name, label, graph_id) DO UPDATE SET
                description = excluded.description,
                embedding = excluded.embedding,
                source_block_id = excluded.source_block_id,
                source_page = excluded.source_page
            """
            params = {
                "name": entity.name,
                "label": entity.label,
                "graph_id": self.graph_id or "default",
                "description": entity.description,
                "embedding": sqlite3.Binary(entity.embedding)
                if entity.embedding is not None
                else None,
                "source_block_id": entity.source_block.block_id if entity.source_block else None,
                "source_page": entity.source_block.page_number if entity.source_block else None,
            }
            self.run_query(query, params)

    def upsert_relations(self, relations: list[Relation]):
        for rel in relations:
            if rel.source_block:
                self._upsert_text_block(rel.source_block)

            query = """
            INSERT INTO relations (start_name, start_label, end_name, end_label, label, name, graph_id, description, embedding, source_block_id, source_page)
            VALUES (:s_name, :s_label, :e_name, :e_label, :label, :name, :graph_id, :description, :embedding, :source_block_id, :source_page)
            ON CONFLICT(start_name, end_name, label, graph_id) DO UPDATE SET
                description = excluded.description,
                embedding = excluded.embedding,
                source_block_id = excluded.source_block_id,
                source_page = excluded.source_page
            """
            params = {
                "s_name": rel.start_entity.name,
                "s_label": rel.start_entity.label,
                "e_name": rel.end_entity.name,
                "e_label": rel.end_entity.label,
                "label": rel.label,
                "name": rel.name,
                "graph_id": self.graph_id or "default",
                "description": rel.description,
                "embedding": sqlite3.Binary(rel.embedding) if rel.embedding is not None else None,
                "source_block_id": rel.source_block.block_id if rel.source_block else None,
                "source_page": rel.source_block.page_number if rel.source_block else None,
            }
            self.run_query(query, params)

    def get_all_entities(self) -> list[Entity]:
        graph_id = self.graph_id or "default"
        rows = self.run_query(
            "SELECT * FROM entities WHERE graph_id = :graph_id", {"graph_id": graph_id}
        )
        entities = []
        for row in rows:
            entities.append(
                Entity(
                    label=row["label"],
                    name=row["name"],
                    embedding=row["embedding"],
                    description=row["description"],
                )
            )
        return entities

    def get_all_relations(self) -> list[Relation]:
        graph_id = self.graph_id or "default"
        rows = self.run_query(
            "SELECT * FROM relations WHERE graph_id = :graph_id", {"graph_id": graph_id}
        )
        relations = []
        for row in rows:
            start_entity = Entity(label=row["start_label"], name=row["start_name"])
            end_entity = Entity(label=row["end_label"], name=row["end_name"])
            relations.append(
                Relation(
                    start_entity=start_entity,
                    end_entity=end_entity,
                    label=row["label"],
                    name=row["name"],
                    embedding=row["embedding"],
                    description=row["description"],
                )
            )
        return relations
