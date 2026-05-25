# 数据读取模块使用说明

## 1. 模块文件

```text
backend/data_reader.py
```


## 2. 主要功能

该模块负责加载 `.h5ad` 单细胞数据文件，提供细胞向量矩阵与细胞元数据的读取接口，供 API 模块（`app.py`）调用。


## 3. 数据集说明

本项目使用的数据集为：

```text
data/liver.h5ad
```

来源：CZI CELLxGENE — *Single-Cell Atlas Of Human Pediatric Liver Reveals Age-Related Hepatic Gene Signatures*

| 字段 | 说明 |
|---|---|
| 细胞数量 | 69,032 |
| 基因数量（`X` 维度） | 33,694 |
| `X` 格式 | CSR 稀疏矩阵，`float64` |
| `obsm/X_pca` | `(69032, 30)`，`float64` |
| `obsm/X_umap` | `(69032, 2)`，`float64` |
| `obsm/X_tsne` | `(69032, 2)`，`float64` |

所有向量由 `DataLoader` 统一转换为 `float32` 输出。

主要可用的 `obs` 元数据字段（细胞信息）：

| 字段名 | 含义 |
|---|---|
| `cell_type` | 细胞类型（如 T cell、hepatocyte） |
| `AgeGroup` | 年龄组 |
| `donor_id` | 捐献者 ID |
| `disease` | 疾病状态（normal / IFALD） |
| `tissue` | 组织来源 |
| `development_stage` | 发育阶段 |
| `sex` | 性别 |
| `donor_age` | 捐献者年龄 |
| `assay` | 测序方式 |
| `author_cell_type` | 作者标注的细胞类型 |


## 4. 接口说明

### 4.1 初始化

```python
from backend.data_reader import DataLoader

loader = DataLoader("data/liver.h5ad")
```

- 文件不存在会抛出 `FileNotFoundError`
- 文件后缀不是 `.h5ad` 会抛出 `ValueError`


### 4.2 `get_vectors(use_rep=None)` — 获取全量向量矩阵

```python
vectors = loader.get_vectors()           # 使用原始基因表达矩阵 X
vectors = loader.get_vectors("X_pca")   # 使用 PCA 降维结果（推荐）
```

**返回值：**

```python
vectors.shape  # (69032, 33694) 或 (69032, 30)
vectors.dtype  # float32
```

- 返回 2D NumPy 数组，可直接传入 `ANNIndexer.build_index(vectors)`
- `use_rep` 可选值：`None`（等同 `"X"`）、`"X_pca"`、`"X_umap"`、`"X_tsne"`
- 传入不存在的 `use_rep` 会抛出 `KeyError`


### 4.3 `get_vector(cell_index, use_rep=None)` — 获取单个细胞向量

```python
query_vector = loader.get_vector(500)
query_vector = loader.get_vector(500, use_rep="X_pca")
```

**返回值：**

```python
query_vector.shape  # (33694,) 或 (30,)
query_vector.dtype  # float32
```

- 返回 1D NumPy 数组，可直接传入 `ANNIndexer.search(query_vector, k)`
- `cell_index` 越界会抛出 `IndexError`
- `cell_index` 类型不是整数会抛出 `TypeError`


### 4.4 `get_cell_info(cell_index)` — 获取细胞元数据

```python
info = loader.get_cell_info(500)
```

**返回值示例：**

```python
{
    "cell_id": "AAACCTGAGCAGGTCA-1_2",
    "cell_type": "hepatocyte",
    "AgeGroup": "pediatric",
    "donor_id": "D1",
    "disease": "normal",
    "tissue": "right lobe of liver",
    "development_stage": "child stage",
    "sex": "male",
    "donor_age": "5",
    ...
}
```

- 返回 Python 原生类型的 `dict`，可直接 `json.dumps()` 序列化
- `cell_index` 越界会抛出 `IndexError`


### 4.5 `vector_dim(use_rep=None)` — 获取向量维度

```python
dim = loader.vector_dim()           # 33694（使用 X）
dim = loader.vector_dim("X_pca")    # 30（使用 X_pca）
```

- 直接用于初始化 `ANNIndexer(dim=loader.vector_dim("X_pca"))`，无需手动计算


### 4.6 便捷属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `loader.n_cells` | `int` | 细胞总数（69032） |
| `loader.n_genes` | `int` | 基因数量（33694） |
| `loader.obs_columns` | `list[str]` | 所有元数据字段名 |
| `loader.available_reps` | `list[str]` | obsm 中可用的降维键名 |


## 5. 在 `app.py` 中的完整调用示例

```python
from backend.data_reader import DataLoader
from backend.ann_indexer import ANNIndexer

# 服务启动时执行一次
USE_REP = "X_pca"
loader = DataLoader("data/liver.h5ad")
vectors = loader.get_vectors(USE_REP)

indexer = ANNIndexer(dim=loader.vector_dim(USE_REP))
indexer.build_index(vectors)

# 每次收到前端请求时执行
def handle_search(cell_index: int, k: int):
    query_vector = loader.get_vector(cell_index, use_rep=USE_REP)
    distances, indices = indexer.search(query_vector, k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices.tolist(), distances.tolist()), start=1):
        cell_info = loader.get_cell_info(idx)
        results.append({
            "rank": rank,
            "cell_index": idx,
            "distance": round(dist, 6),
            "cell_type": cell_info.get("cell_type", "unknown"),
        })

    return {
        "query_cell": cell_index,
        "k": k,
        "results": results,
    }
```


## 6. 异常汇总

| 异常类型 | 触发场景 |
|---|---|
| `FileNotFoundError` | 数据文件路径不存在 |
| `ValueError` | 文件后缀不是 `.h5ad` |
| `KeyError` | `use_rep` 在 `obsm` 中不存在 |
| `IndexError` | `cell_index` 超出 `[0, n_cells-1]` |
| `TypeError` | `cell_index` 不是整数类型 |
