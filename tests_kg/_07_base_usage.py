from disk_kg import DISK
from disk_kg.distiller import Distiller
from disk_kg.provider import ChatProxy, Embeddings, RateLimiter

disk = DISK(
    model=RateLimiter(ChatProxy("config.toml")),
    embedding=Embeddings.build_from("config.toml"),
    enable_token_track=False,
)

file = "tests/test_markdown.md"
file_dst = Distiller.distill(file)

kg = disk.build_knowledge_graph(file_dst)

print(f"Entities(len = {len(kg.entities)}):")
print(list(kg.entities))
print("======================")
print(f"Relations(len = {len(kg.relations)}):")
print(list(kg.relations))
