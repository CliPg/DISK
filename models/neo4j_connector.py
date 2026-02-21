from neo4j import GraphDatabase
from .knowledge_graph import Entity, Relation

class Neo4jConnector:
    """
    A class to connect to a Neo4j database and run queries.
    """
    def __init__(self, uri, user, password):
        """
        Initializes the Neo4jConnector with the given URI, username, and password.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=30, connection_timeout=10)
        self._verify_connectivity()

    def _verify_connectivity(self):
        """Verifies that the connection to Neo4j is working."""
        try:
            self.driver.verify_connectivity()
            print("Neo4j 连接成功")
        except Exception as e:
            print(f"Neo4j 连接失败: {e}")
            raise

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        """
        Runs a Cypher query against the Neo4j database.
        """
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return result.data()

    def create_entities(self, entities:list[Entity]):
        """
        Creates nodes in the Neo4j database for each entity in the list.
        """
        for entity in entities:
            # 使用反引号包裹标签，支持包含空格的标签名
            label = entity.label.replace("`", "``")  # 转义反引号
            query = f"CREATE (n:`{label}` {{name: $name, embedding: $embedding}}) RETURN n"
            parameters = {"name": entity.name, "embedding": entity.embedding}
            self.run_query(query, parameters)

    def create_relations(self, relations:list[Relation]):
        """
        Creates relationships in the Neo4j database for each relation in the list.
        """
        for relation in relations:
            # 使用反引号包裹标签，支持包含空格的标签名
            start_label = relation.start_entity.label.replace("`", "``")
            end_label = relation.end_entity.label.replace("`", "``")
            rel_label = relation.label.replace("`", "``")

            query = f"""
            MATCH (a:`{start_label}` {{name: $start_name}}),
                  (b:`{end_label}` {{name: $end_name}})
            CREATE (a)-[r:`{rel_label}` {{name: $name, embedding: $embedding}}]->(b)
            RETURN a, b, r
            """
            parameters = {
                "start_name": relation.start_entity.name,
                "end_name": relation.end_entity.name,
                "name": relation.name,
                "embedding": relation.embedding
            }
            self.run_query(query, parameters)