from disk_kg import DISK
from disk_kg.models import Entity, KnowledgeGraph, Neo4jConnector, Relation


def create_sample_kg():
    """手动创建一个小的示例知识图谱"""
    # 1. 创建实体
    alice = Entity(label="Person", name="Alice", description="An AI researcher.")
    bob = Entity(label="Person", name="Bob", description="A software engineer.")
    google = Entity(
        label="Company", name="Google", description="A multinational technology company."
    )

    # 2. 创建关系
    # Alice works at Google
    rel1 = Relation(
        start_entity=alice,
        end_entity=google,
        label="WORKS_AT",
        name="works at",
        description="Alice is an AI researcher at Google.",
    )

    # Bob works at Google
    rel2 = Relation(
        start_entity=bob,
        end_entity=google,
        label="WORKS_AT",
        name="works at",
        description="Bob is a software engineer at Google.",
    )

    # Alice knows Bob
    rel3 = Relation(
        start_entity=alice,
        end_entity=bob,
        label="KNOWS",
        name="knows",
        description="Alice and Bob are colleagues.",
    )

    # 3. 构建知识图谱对象
    kg = KnowledgeGraph()
    kg.add_entities([alice, bob, google])
    kg.add_relations([rel1, rel2, rel3])

    return kg


def main():
    # 生成手动知识图谱
    kg = create_sample_kg()

    # Neo4j 配置信息
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "your_neo4j_password"

    print(f"正在连接到 Neo4j: {uri}...")
    try:
        # 使用 Neo4jConnector 存储
        with Neo4jConnector(uri, user, password) as connector:
            print("连接成功，正在存入知识图谱...")
            kg.save_to(connector)
            print("知识图谱存入成功！")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
