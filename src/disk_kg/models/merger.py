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
            
            # Only merge if labels are the same
            if e1.label != e2.label:
                continue

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

    def compact(self, entities: list[Entity], relations: list[Relation]) -> tuple[list[Relation], list[Entity]]:
        """
        Compact a single set of entities and relations by merging similar entities.
        """
        if not entities:
            return relations, entities

        # step1: compute embeddings
        embeddings = np.array([e.embedding for e in entities], dtype=np.float32)
        
        # step2: compute similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized_embeddings = embeddings / norms
        
        sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # step3: find groups using Union-Find
        n = len(entities)
        parent = list(range(n))
        
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
        
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.threshold:
                    if entities[i].label == entities[j].label:
                        union(i, j)
        
        # step4: create merged entities
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
            
        new_entities = []
        entity_map = {} # old_name -> new_entity
        
        for root, indices in groups.items():
            if len(indices) == 1:
                e = entities[indices[0]]
                new_entities.append(e)
                entity_map[e.name] = e
            else:
                # Merge entities in the group
                # Use the one with the shortest name as the canonical name (often more specific or standard)
                # Or just the first one. Let's pick the one with most description.
                best_idx = indices[0]
                max_desc_len = len(entities[best_idx].description or "")
                for idx in indices[1:]:
                    desc_len = len(entities[idx].description or "")
                    if desc_len > max_desc_len:
                        max_desc_len = desc_len
                        best_idx = idx
                
                base_e = entities[best_idx]
                
                # Combine descriptions
                all_descriptions = [entities[idx].description for idx in indices if entities[idx].description]
                unique_descriptions = []
                for d in all_descriptions:
                    if d not in unique_descriptions:
                        unique_descriptions.append(d)
                merged_description = " ".join(unique_descriptions)
                
                # Average embeddings
                group_embeddings = [entities[idx].embedding for idx in indices if entities[idx].embedding is not None]
                if group_embeddings:
                    merged_embedding = np.mean(group_embeddings, axis=0)
                else:
                    merged_embedding = None
                
                merged_entity = Entity(
                    label=base_e.label,
                    name=base_e.name,
                    embedding=merged_embedding,
                    description=merged_description,
                    source_block=base_e.source_block # Just take one source block
                )
                new_entities.append(merged_entity)
                for idx in indices:
                    entity_map[entities[idx].name] = merged_entity
                    
        # step5: update relations
        new_relations = self.update_and_merge_relations(entity_map, relations, [])
        
        return new_relations, new_entities

    def update_and_merge_relations(
        self,
        entity_name_map: dict,
        relations1: list[Relation],
        relations2: list[Relation],
    ) -> list[Relation]:
        """
        Update relations with new entity mappings and merge them (de-duplicate).
        Aggressively merges relations with same start, end and label.
        """
        relation_map = {} # (start_name, end_name, label) -> Relation

        for rel in relations1 + relations2:
            start_entity = entity_name_map.get(rel.start_entity.name, rel.start_entity)
            end_entity = entity_name_map.get(rel.end_entity.name, rel.end_entity)
            
            # Use (start_name, end_name, label) as key for merging relations
            key = (start_entity.name, end_entity.name, rel.label)
            
            if key in relation_map:
                existing_rel = relation_map[key]
                # Combine descriptions and use the one with description if possible
                combined_desc = existing_rel.description
                if rel.description and rel.description not in combined_desc:
                    if combined_desc:
                        combined_desc += " " + rel.description
                    else:
                        combined_desc = rel.description
                
                # Update existing relation properties if new one has them
                # (This is a bit simplified, but good for compaction)
                updated_rel = Relation(
                    start_entity=start_entity,
                    end_entity=end_entity,
                    label=rel.label,
                    name=existing_rel.name, # Keep original name
                    embedding=existing_rel.embedding if existing_rel.embedding is not None else rel.embedding,
                    description=combined_desc,
                    source_block=existing_rel.source_block or rel.source_block
                )
                relation_map[key] = updated_rel
            else:
                updated_rel = Relation(
                    start_entity=start_entity,
                    end_entity=end_entity,
                    label=rel.label,
                    name=rel.name,
                    embedding=rel.embedding,
                    description=rel.description,
                    source_block=rel.source_block
                )
                relation_map[key] = updated_rel

        return list(relation_map.values())
