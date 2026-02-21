## cypher
1. 查看所有实体和关系
```
MATCH (a)-[r]->(b) RETURN a, r, b
```

2. 删除所有实体和关系
```
MATCH (n) DETACH DELETE n
```