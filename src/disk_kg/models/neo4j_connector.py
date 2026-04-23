from typing import Any, LiteralString, cast

from neo4j import GraphDatabase

from disk_kg.distiller import TextBlock

from .connector import Connector
from .knowledge_graph import Entity, Relation


class Neo4jConnector(Connector):
    """
    A class to connect to a Neo4j database and run queries.
    """

    def __init__(self, uri, user, password, graph_id: str | None = None):
        """
        Initializes the Neo4jConnector with the given URI, username, and password.
        """
        super().__init__(graph_id)
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), max_connection_lifetime=30, connection_timeout=10
        )
        self._verify_connectivity()

    def _verify_connectivity(self):
        """Verifies that the connection to Neo4j is working."""
        try:
            self.driver.verify_connectivity()
            print("Neo4j 连接成功")
        except Exception as e:
            print(f"Neo4j 连接失败: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters=None) -> Any:
        """
        Runs a Cypher query against the Neo4j database.
        """
        with self.driver.session() as session:
            result = session.run(cast(LiteralString, query), parameters)
            return result.data()

    def _upsert_text_block(self, block: TextBlock):
        """Internal helper to upsert a TextBlock node."""
        query = """
        MERGE (b:TextBlock {block_id: $block_id, page_number: $page_number, graph_id: $graph_id})
        SET b.text = $text, b.file_path = $file_path
        RETURN b
        """
        parameters = {
            "block_id": block.block_id,
            "page_number": block.page_number,
            "text": block.text,
            "graph_id": self.graph_id or "default",
        }
        self.run_query(query, parameters)

    def upsert_entities(self, entities: list[Entity]):
        """
        Creates/updates nodes in the Neo4j database for each entity.
        Links entities to their source TextBlock if available.
        """
        for entity in entities:
            label = entity.label.replace("`", "``")
            graph_id = self.graph_id or "default"

            # MERGE entity
            query = f"""
            MERGE (n:`{label}` {{name: $name, graph_id: $graph_id}})
            SET n.embedding = $embedding, n.description = $description
            """
            params = {
                "name": entity.name,
                "embedding": entity.embedding,
                "description": entity.description,
                "graph_id": graph_id,
            }
            self.run_query(query, params)

            # Link to source block if exists
            if entity.source_block:
                self._upsert_text_block(entity.source_block)
                link_query = f"""
                MATCH (n:`{label}` {{name: $name, graph_id: $graph_id}})
                MATCH (b:TextBlock {{block_id: $block_id, page_number: $page_number, graph_id: $graph_id}})
                MERGE (n)-[:MENTIONED_IN]->(b)
                """
                link_params = {
                    "name": entity.name,
                    "graph_id": graph_id,
                    "block_id": entity.source_block.block_id,
                    "page_number": entity.source_block.page_number,
                }
                self.run_query(link_query, link_params)

    def upsert_relations(self, relations: list[Relation]):
        """
        Creates/updates relationships in the Neo4j database for each relation.
        Links relations to their source TextBlock if available.
        """
        for relation in relations:
            start_label = relation.start_entity.label.replace("`", "``")
            end_label = relation.end_entity.label.replace("`", "``")
            rel_label = relation.label.replace("`", "``")
            graph_id = self.graph_id or "default"

            query = f"""
            MATCH (a:`{start_label}` {{name: $start_name, graph_id: $graph_id}})
            MATCH (b:`{end_label}` {{name: $end_name, graph_id: $graph_id}})
            MERGE (a)-[r:`{rel_label}`]->(b)
            SET r.name = $name, r.embedding = $embedding, r.description = $description, r.graph_id = $graph_id
            """
            params = {
                "start_name": relation.start_entity.name,
                "end_name": relation.end_entity.name,
                "name": relation.name,
                "embedding": relation.embedding,
                "description": relation.description,
                "graph_id": graph_id,
            }
            self.run_query(query, params)

            # Link relationship to source block
            if relation.source_block:
                self._upsert_text_block(relation.source_block)
                update_rel_query = f"""
                MATCH (a:`{start_label}` {{name: $start_name, graph_id: $graph_id}})-[r:`{rel_label}`]->(b:`{end_label}` {{name: $end_name, graph_id: $graph_id}})
                SET r.source_block_id = $block_id, r.source_page = $page_number
                """
                self.run_query(
                    update_rel_query,
                    {
                        "start_name": relation.start_entity.name,
                        "end_name": relation.end_entity.name,
                        "graph_id": graph_id,
                        "block_id": relation.source_block.block_id,
                        "page_number": relation.source_block.page_number,
                    },
                )

    def get_all_entities(self) -> list[Entity]:
        """
        Retrieves all entities from Neo4j (for the current graph_id).
        """
        graph_id = self.graph_id or "default"
        query = "MATCH (n) WHERE n.graph_id = $graph_id AND NOT n:TextBlock RETURN properties(n) as n, labels(n) as labels"
        results = self.run_query(query, {"graph_id": graph_id})

        entities = []
        for record in results:
            node = record["n"]
            labels = record["labels"]
            if labels:
                entities.append(
                    Entity(
                        label=labels[0],
                        name=node.get("name", ""),
                        embedding=node.get("embedding"),
                        description=node.get("description", ""),
                    )
                )
        return entities

    def get_all_relations(self) -> list[Relation]:
        """
        Retrieves all relations from Neo4j (for the current graph_id).
        """
        graph_id = self.graph_id or "default"
        query = """
        MATCH (a)-[r]->(b)
        WHERE r.graph_id = $graph_id AND NOT a:TextBlock AND NOT b:TextBlock
        RETURN properties(a) as a, labels(a) as a_labels, properties(b) as b, labels(b) as b_labels, properties(r) as r, type(r) as r_type
        """
        results = self.run_query(query, {"graph_id": graph_id})

        relations = []
        for record in results:
            a_node = record["a"]
            b_node = record["b"]
            rel = record["r"]

            start_entity = Entity(label=record["a_labels"][0], name=a_node.get("name", ""))
            end_entity = Entity(label=record["b_labels"][0], name=b_node.get("name", ""))

            relations.append(
                Relation(
                    start_entity=start_entity,
                    end_entity=end_entity,
                    label=record["r_type"],
                    name=rel.get("name", ""),
                    embedding=rel.get("embedding"),
                    description=rel.get("description", ""),
                )
            )
        return relations

    def clear(self, confirm: str | None = None) -> None:
        """
        Clears all data from the Neo4j database for the current graph_id.
        """
        if not self._confirm_clear(confirm):
            return

        graph_id = self.graph_id or "default"
        query = "MATCH (n) WHERE n.graph_id = $graph_id DETACH DELETE n"
        self.run_query(query, {"graph_id": graph_id})
        print(f"图 '{graph_id}' 的数据已清除。")
