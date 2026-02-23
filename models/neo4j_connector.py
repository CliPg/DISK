from neo4j import GraphDatabase
from .knowledge_graph import Entity, Relation

class Neo4jConnector:
    """
    A class to connect to a Neo4j database and run queries.
    """
    def __init__(self, uri, user, password, graph_id: str = None):
        """
        Initializes the Neo4jConnector with the given URI, username, and password.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            graph_id: Knowledge graph ID for data isolation
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=30, connection_timeout=10)
        self.graph_id = graph_id
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
        Each node will be tagged with graph_id for data isolation.
        """
        for entity in entities:
            # 使用反引号包裹标签，支持包含空格的标签名
            label = entity.label.replace("`", "``")  # 转义反引号

            # 构建 SET 子句，包含 graph_id 和其他属性
            set_clause = "name: $name, embedding: $embedding"
            if self.graph_id:
                set_clause += ", graph_id: $graph_id"

            query = f"CREATE (n:`{label}` {{{set_clause}}}) RETURN n"
            parameters = {"name": entity.name, "embedding": entity.embedding}
            if self.graph_id:
                parameters["graph_id"] = self.graph_id
            self.run_query(query, parameters)

    def create_relations(self, relations:list[Relation]):
        """
        Creates relationships in the Neo4j database for each relation in the list.
        Each relationship will be tagged with graph_id for data isolation.
        """
        for relation in relations:
            # 使用反引号包裹标签，支持包含空格的标签名
            start_label = relation.start_entity.label.replace("`", "``")
            end_label = relation.end_entity.label.replace("`", "``")
            rel_label = relation.label.replace("`", "``")

            # 构建 SET 子句，包含 graph_id 和其他属性
            set_clause = "name: $name, embedding: $embedding"
            if self.graph_id:
                set_clause += ", graph_id: $graph_id"

            query = f"""
            MATCH (a:`{start_label}` {{name: $start_name}}),
                  (b:`{end_label}` {{name: $end_name}})
            CREATE (a)-[r:`{rel_label}` {{{set_clause}}}]->(b)
            RETURN a, b, r
            """
            parameters = {
                "start_name": relation.start_entity.name,
                "end_name": relation.end_entity.name,
                "name": relation.name,
                "embedding": relation.embedding
            }
            if self.graph_id:
                parameters["graph_id"] = self.graph_id
            self.run_query(query, parameters)