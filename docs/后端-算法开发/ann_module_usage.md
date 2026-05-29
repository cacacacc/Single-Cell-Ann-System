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
	index_type="hnsw",      # flat / ivf_flat / hnsw / brute
	metric="cosine",        # l2 / cosine / ip
	m=32,
	ef_construction=200,
	ef_search=50,
)

indexer = ANNIndexer(dim=50, config=config)
indexer.build_index(vectors)

query_vector = vectors[0]
distances, indices = indexer.search(query_vector, k=10)

print(indices)
print(distances)
```


## 4. 输入要求

传入的向量矩阵必须满足：

```python
vectors.dtype == np.float32
len(vectors.shape) == 2
```


其中：

- 第一维表示细胞数量
    
- 第二维表示向量维度
    

## 5. 输出说明

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
    

## 6. 与 API 模块的对接方式

API 开发人员可以在 `app.py` 中这样调用：

```python
query_vector = data_loader.get_vector(cell_index)
distances, indices = indexer.search(query_vector, k)
```