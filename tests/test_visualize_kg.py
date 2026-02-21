from disk import DISK
from config.llm import llm, embeddings
import os
import sys
from pathlib import Path
ROOT = Path().resolve().parent
sys.path.append(str(ROOT))
from extractor import Extractor
from merger import Merger
from config.llm import llm, embeddings
from manager.kg_manager import KGManager
from models.knowledge_graph import Entity, Relation
disk = DISK(llm=llm, embeddings=embeddings)

entities1 = {"entities": [
    {"label": "Person", "name": "李明"},
    {"label": "Organization", "name": "北京航空航天大学计算机学院"},
    {"label": "Academic Rank", "name": "副教授"},
    {"label": "Research Field", "name": "人工智能"},
    {"label": "Organization", "name": "清华大学"}]}

entities2 = {"entities": [
    {"label": "Person", "name": "李明"},
    {"label": "Organization", "name": "Tsinghua University"},
    {"label": "AcademicDegree", "name": "PhD"},
    {"label": "ResearchField", "name": "natural language understanding"},
    {"label": "ResearchField", "name": "deep learning"},
    {"label": "ResearchField", "name": "knowledge graph construction"},
    {"label": "EvaluationTask", "name": "NLP evaluation tasks"}]}

relations1 = {"relations": [
    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "Organization", "name": "北京航空航天大学计算机学院"},
    "label": "affiliation", "name": "is affiliated with"},

    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "Academic Rank", "name": "副教授"},
    "label": "has_academic_rank", "name": "holds the position of"},

    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "Research Field", "name": "人工智能"},
    "label": "researches_in", "name": "conducts research in"},

    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "Organization", "name": "清华大学"},
    "label": "earned_phd_from", "name": "earned PhD from"}]}

relations2 = {"relations": [
    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "Organization", "name": "Tsinghua University"},
    "label": "affiliated_with", "name": "affiliation"},

    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "AcademicDegree", "name": "PhD"},
    "label": "holds_degree", "name": "doctoral degree"},

    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "ResearchField", "name": "natural language understanding"},
    "label": "researches_in", "name": "research focus"},

    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "ResearchField", "name": "deep learning"},
    "label": "researches_in", "name": "research focus"},

    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "ResearchField", "name": "knowledge graph construction"},
    "label": "researches_in", "name": "research focus"},

    {"start_entity": {"label": "Person", "name": "李明"},
    "end_entity": {"label": "EvaluationTask", "name": "NLP evaluation tasks"},
    "label": "achieves_performance_in", "name": "performance achievement"}]}

extractor = Extractor(llm=llm, embeddings=embeddings)
print("正在为实体生成嵌入...")
emb_entities1 = extractor.embed_entities(entities1)
emb_entities2 = extractor.embed_entities(entities2)
print("正在为关系生成嵌入...")
emb_relations1 = extractor.embed_relations(relations1)
emb_relations2 = extractor.embed_relations(relations2)

merger = Merger(threshold=0.65)
print("正在合并知识图谱...")
merged_relations, merged_entities = merger.merge(emb_entities1, emb_entities2, emb_relations1, emb_relations2)

uri = "neo4j://127.0.0.1:7687"
user = "neo4j"
password = "12345678"
print(f"正在连接 Neo4j ({uri})...")
disk.visualize_knowledge_graph(uri=uri, user=user, password=password, entities=merged_entities, relations=merged_relations)
print("知识图谱可视化完成！")
