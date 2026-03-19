import os

import numpy as np

from disk_kg.models import Entity, Relation


class Merger:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        # Set log directory to project root/logs
        self.log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def merge(
        self,
        entities1: list[Entity],
        entities2: list[Entity],
        relations1: list[Relation],
        relations2: list[Relation],
    ) -> tuple[list[Relation], list[Entity]]:
        # step1: compute embeddings
        # 将 list 转换为 numpy array (N, D) 和 (M, D)
        emb1 = np.array([e.embedding for e in entities1], dtype=np.float32)
        emb2 = np.array([e.embedding for e in entities2], dtype=np.float32)

        # step2: compute similarity matrix
        # 计算向量的 L2 范数 (N, 1) 和 (M, 1)
        norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)

        # 避免分母为 0 的情况
        norm1 = np.maximum(norm1, 1e-8)
        norm2 = np.maximum(norm2, 1e-8)

        # 矩阵乘法计算点积，并除以范数得到余弦相似度 (N, M)
        # 公式: (A @ B.T) / (|A| * |B|)
        sim_matrix = np.dot(emb1, emb2.T) / np.dot(norm1, norm2.T)

        with open(os.path.join(self.log_dir, "sim_matrix.log"), "a") as f:
            f.write(f"Similarity Matrix:\n{sim_matrix}\n")

        with open(os.path.join(self.log_dir, "entities.log"), "a") as f:
            f.write("Entities1:\n")
            for e in entities1:
                f.write(f"{e.name}\n")
            f.write("Entities2:\n")
            for e in entities2:
                f.write(f"{e.name}\n")

        # step3: find matches above threshold
        matches = np.where(sim_matrix >= self.threshold)

        merged_entities = []
        used_2 = set()
        used_1 = set()

        entity_name_map = {}

        for idx1, idx2 in zip(matches[0].tolist(), matches[1].tolist()):
            e1 = entities1[idx1]
            e2 = entities2[idx2]
            used_2.add(idx2)
            used_1.add(idx1)  # 标记 entities1 中已合并的索引

            merged_entity = Entity(
                name=e1.name,  # 或者合并名字
                label=e1.label,
                embedding=emb1[idx1].tolist(),
                description=e1.description if hasattr(e1, "description") else "",
            )

            entity_name_map[e1.name] = merged_entity
            entity_name_map[e2.name] = merged_entity

            merged_entities.append(merged_entity)

        # 4. 对于 entities2 里未被合并的实体，保留
        for i, e2 in enumerate(entities2):
            if i not in used_2:
                merged_entities.append(e2)

        # 5. 再加上 entities1 中没有匹配到的实体
        for i, e1 in enumerate(entities1):
            if i not in used_1:
                merged_entities.append(e1)

        with open(os.path.join(self.log_dir, "merged_entities.log"), "a") as f:
            for idx1, idx2 in zip(matches[0].tolist(), matches[1].tolist()):
                e1 = entities1[idx1]
                e2 = entities2[idx2]
                f.write(f"Merged: {e1.name} + {e2.name}\n")

        merged_relations = self.update_and_merge_relations(entity_name_map, relations1, relations2)

        with open(os.path.join(self.log_dir, "merged_relations.log"), "a") as f:
            for rel in merged_relations:
                f.write(
                    f"Merged relations: {rel.start_entity.name} -[{rel.name}]-> {rel.end_entity.name}\n"
                )

        return merged_relations, merged_entities

    def update_and_merge_relations(
        self,
        entity_name_map: dict,
        relations1: list[Relation] = [],
        relations2: list[Relation] = [],
    ) -> tuple[list[Relation], list[Entity]]:
        relations = []

        for rel in relations1:
            start_entity = entity_name_map.get(rel.start_entity.name, rel.start_entity)
            end_entity = entity_name_map.get(rel.end_entity.name, rel.end_entity)
            updated_relation = Relation(
                start_entity=start_entity,
                end_entity=end_entity,
                label=rel.label,
                name=rel.name,
                embedding=rel.embedding,
                description=rel.description if hasattr(rel, "description") else "",
            )
            relations.append(updated_relation)

        for rel in relations2:
            start_entity = entity_name_map.get(rel.start_entity.name, rel.start_entity)
            end_entity = entity_name_map.get(rel.end_entity.name, rel.end_entity)
            updated_relation = Relation(
                start_entity=start_entity,
                end_entity=end_entity,
                label=rel.label,
                name=rel.name,
                embedding=rel.embedding,
                description=rel.description if hasattr(rel, "description") else "",
            )
            relations.append(updated_relation)

        return relations
