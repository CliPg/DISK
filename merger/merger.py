import torch
import torch.nn.functional as F
from models import Entity


class Merger:
    def __init__(self, entities1: list[Entity], entities2: list[Entity], 
                relations1: list, relations2: list,
                threshold:float=0.8):
        
        self.threshold = threshold
        self.entities1 = entities1
        self.entities2 = entities2
        self.relations1 = relations1
        self.relations2 = relations2

    def merge(self) -> list[Entity]:
        entities1 = self.entities1
        entities2 = self.entities2

        # step1: compute embeddings
        emb1 = torch.stack([torch.tensor(e.embedding) for e in entities1])   # (N, D)
        emb2 = torch.stack([torch.tensor(e.embedding) for e in entities2])   # (M, D)

        # step2: compute similarity matrix
        sim_matrix = F.cosine_similarity(
            emb1[:, None, :],   # (N, 1, D)
            emb2[None, :, :],   # (1, M, D)
            dim=-1
        )

        with open("../logs/sim_matrix.log", "a") as f:
            f.write(f"Similarity Matrix:\n{sim_matrix}\n")

        with open("../logs/entities.log", "a") as f:
            f.write(f"Entities1:\n")
            for e in entities1:
                f.write(f"{e.name}\n")
            f.write(f"Entities2:\n")
            for e in entities2:
                f.write(f"{e.name}\n")

        # step3: find matches above threshold
        matches = torch.where(sim_matrix >= self.threshold)

        merged = []
        used_2 = set()

        entity_name_map = {}

        for idx1, idx2 in zip(matches[0].tolist(), matches[1].tolist()):
            e1 = entities1[idx1]
            e2 = entities2[idx2]
            used_2.add(idx2)

            # 创建新的 merged entity（可自定义，例如取平均 embedding）
            merged_entity = Entity(
                name=e1.name,     # 或者合并名字
                label=e1.label,
                embedding=emb1[idx1].tolist()
            )

            entity_name_map[e1.name] = merged_entity
            entity_name_map[e2.name] = merged_entity

            merged.append(merged_entity)

        # 4. 对于 entities2 里未被合并的实体，保留
        for i, e2 in enumerate(entities2):
            if i not in used_2:
                merged.append(e2)

        # 5. 再加上 entities1 中没有匹配到的实体
        matched_1 = set(matches[0].tolist())
        for i, e1 in enumerate(entities1):
            if i not in matched_1:
                merged.append(e1)

        with open("../logs/merged_entities.log", "a") as f:
            for idx1, idx2 in zip(matches[0].tolist(), matches[1].tolist()):
                e1 = entities1[idx1]
                e2 = entities2[idx2]
                f.write(f"Merged: {e1.name} + {e2.name}\n")

        return merged
    
    def update_relations(self, entity_name_map: dict) -> tuple[list, list]:
        updated_relations1 = []
        updated_relations2 = []

        for rel in self.relations1:
            start_entity = entity_name_map.get(rel.start_entity.name, rel.start_entity)
            end_entity = entity_name_map.get(rel.end_entity.name, rel.end_entity)
            updated_relation = type(rel)(
                start_entity=start_entity,
                end_entity=end_entity,
                label=rel.label,
                name=rel.name,
                embedding=rel.embedding
            )
            updated_relations1.append(updated_relation)

        for rel in self.relations2:
            start_entity = entity_name_map.get(rel.start_entity.name, rel.start_entity)
            end_entity = entity_name_map.get(rel.end_entity.name, rel.end_entity)
            updated_relation = type(rel)(
                start_entity=start_entity,
                end_entity=end_entity,
                label=rel.label,
                name=rel.name,
                embedding=rel.embedding
            )
            updated_relations2.append(updated_relation)

        return updated_relations1, updated_relations2
    
