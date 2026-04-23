import time

from disk_kg.models import KnowledgeGraph, Neo4jConnector

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_neo4j_password"  # 请替换为实际密码
SOURCE_GRAPH = "default"
TARGET_GRAPH = "compacted_prod_v1"


def production_compact_benchmark():
    # 配置 Neo4j 连接（建议通过环境变量或配置文件管理密码）
    kg = KnowledgeGraph()

    print("--- 开始生产环境压缩测试 ---")
    print(f"目标数据库: {URI}")

    with Neo4jConnector(URI, USER, PASSWORD, graph_id=SOURCE_GRAPH) as connector:
        # 1. 加载数据
        print(f"正在从 '{SOURCE_GRAPH}' 加载数据...")
        start_time = time.time()
        kg.load_from(connector)
        load_duration = time.time() - start_time

        initial_entities = len(kg.entities)
        initial_relations = len(kg.relations)
        print(
            f"加载完成: {initial_entities} 节点, {initial_relations} 关系 (耗时: {load_duration:.2f}s)"
        )

        if initial_entities == 0:
            print("错误: 未发现可压缩的节点，请确认 graph_id 或数据库内容。")
            return

        # 2. 执行压缩
        print("正在执行内存压缩 (阈值=0.85)...")
        start_time = time.time()
        kg.compact(threshold=0.85)
        compact_duration = time.time() - start_time

        final_entities = len(kg.entities)
        final_relations = len(kg.relations)

        print("压缩完成!")
        print(
            f"  - 节点变化: {initial_entities} -> {final_entities} (减少 {(initial_entities - final_entities) / initial_entities * 100:.1f}%)"
        )
        print(f"  - 关系变化: {initial_relations} -> {final_relations}")
        print(f"  - 压缩算法耗时: {compact_duration:.2f}s")

        # 3. 持久化存储 (安全起见，写入新图)
        print(f"正在将压缩后的数据写入新图 '{TARGET_GRAPH}'...")
        start_time = time.time()
        with Neo4jConnector(URI, USER, PASSWORD, graph_id=TARGET_GRAPH) as target_conn:
            target_conn.clear()  # 清理目标图
            kg.save_to(target_conn)
        save_duration = time.time() - start_time
        print(f"存储完成 (耗时: {save_duration:.2f}s)")

        print("\n--- 总结报告 ---")
        print(f"总处理规模: {initial_entities + initial_relations} 元素")
        print(f"总耗时: {load_duration + compact_duration + save_duration:.2f}s")


if __name__ == "__main__":
    production_compact_benchmark()
