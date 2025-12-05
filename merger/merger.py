import torch
import torch.nn.functional as F
from models import Entity, Relation
from utils.schemas import EntitySchema


class Merger:
    def __init__(self, entities1: list[Entity]=[], entities2: list[Entity]=[], 
                relations1: list[Relation]=[], relations2: list[Relation]=[],
                threshold:float=0.8):
        
        self.threshold = threshold

    def merge(self, entities1: list[Entity]=[], entities2: list[Entity]=[], 
                relations1: list[Relation]=[], relations2: list[Relation]=[]) -> tuple[list[Relation], list[Entity]]:

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

        merged_entities = []
        used_2 = set()

        entity_name_map = {}

        for idx1, idx2 in zip(matches[0].tolist(), matches[1].tolist()):
            e1 = entities1[idx1]
            e2 = entities2[idx2]
            used_2.add(idx2)

            merged_entity = Entity(
                name=e1.name,     # 或者合并名字
                label=e1.label,
                embedding=emb1[idx1].tolist()
            )

            entity_name_map[e1.name] = merged_entity
            entity_name_map[e2.name] = merged_entity

            merged_entities.append(merged_entity)

        # 4. 对于 entities2 里未被合并的实体，保留
        for i, e2 in enumerate(entities2):
            if i not in used_2:
                merged_entities.append(e2)

        # 5. 再加上 entities1 中没有匹配到的实体
        matched_1 = set(matches[0].tolist())
        for i, e1 in enumerate(entities1):
            if i not in matched_1:
                merged_entities.append(e1)

        with open("../logs/merged_entities.log", "a") as f:
            for idx1, idx2 in zip(matches[0].tolist(), matches[1].tolist()):
                e1 = entities1[idx1]
                e2 = entities2[idx2]
                f.write(f"Merged: {e1.name} + {e2.name}\n")

        merged_relations = self.update_and_merge_relations(entity_name_map, relations1, relations2)

        with open("../logs/merged_relations.log", "a") as f:
            for rel in merged_relations:
                f.write(f"Merged relations: {rel.start_entity.name} -[{rel.name}]-> {rel.end_entity.name}\n")

        return merged_relations, merged_entities
    
    def update_and_merge_relations(self, entity_name_map: dict, relations1: list[Relation]=[], relations2: list[Relation]=[]) -> tuple[list[Relation], list[Entity]]:
        
        relations = []

        for rel in relations1:
            start_entity = entity_name_map.get(rel.start_entity.name, rel.start_entity)
            end_entity = entity_name_map.get(rel.end_entity.name, rel.end_entity)
            updated_relation = Relation(
                start_entity=start_entity,
                end_entity=end_entity,
                label=rel.label,
                name=rel.name,
                embedding=rel.embedding
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
                embedding=rel.embedding
            )
            relations.append(updated_relation)

        return relations
    
