# ANN 检索算法设计说明

## 1. 模块目标

本模块负责接收数据读取模块提供的细胞向量矩阵，构建近似最近邻索引，并支持根据查询细胞向量返回 Top-K 相似细胞结果。

## 2. 输入与输出

### 输入

- 细胞向量矩阵：`vectors`
- 数据格式：`numpy.ndarray`
- 形状：`(cell_count, vector_dim)`
- 数据类型：`float32`

示例：

```python
vectors.shape == (10000, 50)
vectors.dtype == np.float32
``

### 输出

检索结果包括：

- 相似细胞编号
    
- 距离或相似度
    
- 排名 rank
    

示例：

```json
[
  {
    "rank": 1,
    "cell_index": 128,
    "distance": 0.0321
  }
]
```

## 3. 算法选择

当前已实现：

- FAISS：`IndexFlatL2` / `IndexFlatIP`、`IVFFlat`、`HNSWFlat`、`PQ`
- HNSWLIB：`HNSW`（`l2`/`cosine`/`ip`）
- NumPy：暴力检索（作为 recall 基准）

支持的距离度量：

- `l2`（默认，返回平方 L2 距离）
- `cosine`（返回 $1-\cos$ 距离）
- `ip`（返回 $-\langle q, v \rangle$ 距离）

可配置参数（通过环境变量或 API 参数传入）：

- `nlist` / `nprobe`（IVF）
- `M` / `ef_construction` / `ef_search`（HNSW）
- `pq_m` / `pq_nbits`（PQ，要求 `pq_m` 能整除向量维度）
    

## 4. 核心流程

```text
接收细胞向量矩阵
        ↓
初始化 ANN 索引
        ↓
添加所有细胞向量
        ↓
输入查询细胞向量
        ↓
执行 Top-K 检索
        ↓
返回相似细胞编号和距离
```

## 5. 类设计

```python
class ANNIndexer:
    def __init__(self, dim):
        pass

    def build_index(self, vectors):
        pass

    def search(self, query_vector, k):
        pass

    def save_index(self, index_path):
        pass

    def load_index(self, index_path):
        pass
```

## 6. 异常处理

需要考虑：

- 输入向量为空
    
- 查询细胞编号越界
    
- K 值非法
    
- 向量维度不匹配
    
- 索引尚未构建就执行查询
    

## 7. 后续优化方向

- 支持索引缓存
- 支持多种距离度量
- 支持不同 ANN 算法对比
- 支持查询耗时统计
- 支持召回率评估
    