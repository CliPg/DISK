import numpy as np

from .knowledge_graph import Entity, Relation


class Merger:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def merge(
        self,
        entities1: list[Entity],
        entities2: list[Entity],
        relations1: list[Relation],
        relations2: list[Relation],
    ) -> tuple[list[Relation], list[Entity]]:
        """
        Merge two sets of entities and relations based on entity similarity.
        """
        if not entities1:
            return relations2, entities2
        if not entities2:
            return relations1, entities1

        # step1: compute embeddings
        emb1 = np.array([e.embedding for e in entities1], dtype=np.float32)
        emb2 = np.array([e.embedding for e in entities2], dtype=np.float32)

        # step2: compute similarity matrix
        norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)

        norm1 = np.maximum(norm1, 1e-8)
        norm2 = np.maximum(norm2, 1e-8)

        sim_matrix = np.dot(emb1, emb2.T) / np.dot(norm1, norm2.T)

        # step3: find matches above threshold
        matches = np.where(sim_matrix >= self.threshold)

        merged_entities = []
        used_2 = set()
        used_1 = set()

        entity_name_map = {}

        for idx1, idx2 in zip(matches[0].tolist(), matches[1].tolist()):
            # Handle potential multiple matches - just take the first one or skip if already used
            if idx1 in used_1 or idx2 in used_2:
                continue

            e1 = entities1[idx1]
            e2 = entities2[idx2]
            used_2.add(idx2)
            used_1.add(idx1)

            # Create merged entity (prioritize e1 properties)
            merged_entity = Entity(
                name=e1.name,
                label=e1.label,
                embedding=e1.embedding,
                description=e1.description or e2.description,
                source_block=e1.source_block or e2.source_block
            )

            entity_name_map[e1.name] = merged_entity
            entity_name_map[e2.name] = merged_entity

            merged_entities.append(merged_entity)

        # Add remaining entities from entities2
        for i, e2 in enumerate(entities2):
            if i not in used_2:
                merged_entities.append(e2)

        # Add remaining entities from entities1
        for i, e1 in enumerate(entities1):
            if i not in used_1:
                merged_entities.append(e1)

        merged_relations = self.update_and_merge_relations(entity_name_map, relations1, relations2)

        return merged_relations, merged_entities

    def update_and_merge_relations(
        self,
        entity_name_map: dict,
        relations1: list[Relation],
        relations2: list[Relation],
    ) -> list[Relation]:
        """
        Update relations with new entity mappings and merge them (de-duplicate).
        """
        unique_relations = set()

        for rel in relations1 + relations2:
            start_entity = entity_name_map.get(rel.start_entity.name, rel.start_entity)
            end_entity = entity_name_map.get(rel.end_entity.name, rel.end_entity)
            
            updated_relation = Relation(
                start_entity=start_entity,
                end_entity=end_entity,
                label=rel.label,
                name=rel.name,
                embedding=rel.embedding,
                description=rel.description,
                source_block=rel.source_block
            )
            unique_relations.add(updated_relation)

        return list(unique_relations)
