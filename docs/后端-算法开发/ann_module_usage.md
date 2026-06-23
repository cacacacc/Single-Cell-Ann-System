# ANN 模块使用说明

## 1. 模块文件

```text
backend/ann_indexer.py
```


## 2. 主要功能

该模块提供 ANN 索引构建、Top-K 检索、索引保存和索引加载功能。

## 3. 使用示例

```python
import numpy as np
from ann_indexer import ANNIndexer, IndexConfig

vectors = np.random.random((1000, 50)).astype("float32")

# 通过配置选择后端、索引类型与度量
config = IndexConfig(
	backend="auto",        # auto / faiss / hnswlib / numpy
	index_type="hnsw",      # flat / ivf_flat / hnsw / pq / brute
	metric="cosine",        # l2 / cosine / ip
	m=32,
	ef_construction=200,
	ef_search=50,
	pq_m=8,
	pq_nbits=8,
)

indexer = ANNIndexer(dim=50, config=config)
indexer.build_index(vectors)

query_vector = vectors[0]
distances, indices = indexer.search(query_vector, k=10)

print(indices)
print(distances)
```

PQ 参数说明：

- `pq_m`：子空间数量（需整除向量维度）
- `pq_nbits`：每个子空间编码位数（常用 8）


## 4. 索引保存与加载

```python
# 保存索引到文件
indexer.save_index("indexes/my_index.index", use_rep="X_pca")

# 从文件加载索引
indexer = ANNIndexer(dim=50)
indexer.load_index("indexes/my_index.index")

# 加载后可直接检索
distances, indices = indexer.search(query_vector, k=10)
```

索引保存时会同时生成 `.npz` 归档文件（包含向量数据和完整配置 JSON），加载时优先从 `.npz` 读取，保证跨平台兼容性。

## 5. 从环境变量构建配置

```python
# 从环境变量读取索引配置（前缀 CELL_INDEX_）
config = IndexConfig.from_env()

# 等价于读取以下环境变量：
# CELL_INDEX_BACKEND=auto
# CELL_INDEX_TYPE=flat
# CELL_INDEX_METRIC=l2
# CELL_INDEX_NLIST=100
# CELL_INDEX_NPROBE=10
# CELL_INDEX_M=16
# CELL_INDEX_EF_CONSTRUCTION=200
# CELL_INDEX_EF_SEARCH=50
# CELL_INDEX_PQ_M=8
# CELL_INDEX_PQ_NBITS=8
```

也可以在 API 请求参数中动态传入覆盖默认值。

## 6. 配置摘要属性

```python
print(indexer.config_summary)
# {
#   "backend": "faiss",
#   "index_type": "hnsw",
#   "metric": "l2",
#   "m": 16,
#   "ef_construction": 200,
#   "ef_search": 50
# }
```

`config_summary` 返回当前索引的实际配置（包括构建时确定的后端和参数），常用于 API 响应中展示索引信息。

## 7. 输入要求

传入的向量矩阵必须满足：

```python
vectors.dtype == np.float32
len(vectors.shape) == 2
```


其中：

- 第一维表示细胞数量
    
- 第二维表示向量维度
    

## 8. 输出说明

`search()` 方法返回两个结果：

```python
distances, indices
```


其中：

- `distances`：每个相似细胞与查询细胞的距离
    
- `indices`：相似细胞在原始数据中的编号

距离含义：

- `l2`：平方 L2 距离（越小越相似）
- `cosine`：$1-\cos$ 距离（越小越相似）
- `ip`：$-\langle q, v \rangle$（越小越相似）
    

## 9. 与 API 模块的对接方式

API 开发人员可以在 `app.py` 中这样调用：

```python
query_vector = data_loader.get_vector(cell_index)
distances, indices = indexer.search(query_vector, k)
```