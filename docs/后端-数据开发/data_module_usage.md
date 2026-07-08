# 数据读取模块使用说明

## 1. 模块文件

```text
backend/data_reader.py
```

本模块提供两个核心类：

- **`DataLoader`**：负责加载单个 `.h5ad` 文件，提供向量矩阵与细胞元数据读取接口
- **`DatasetManager`**：多数据集管理器，支持数据集的增删查、上传校验和元信息维护


## 2. 数据集说明

本项目默认数据集为：

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

`var` 基因信息用于高表达基因索引。当前 `DataLoader` 会保留 `var_names` 和 `var` 列名，并在需要展示基因符号时优先从 `feature_name`、`gene_symbol`、`gene_name`、`symbol`、`name` 等列中读取；若这些列不存在，则回退使用 `var_names`。


---

## 3. DataLoader 接口说明

### 3.1 初始化

```python
from backend.data_reader import DataLoader

loader = DataLoader("data/liver.h5ad")
```

- 文件不存在会抛出 `FileNotFoundError`
- 文件后缀不是 `.h5ad` 会抛出 `ValueError`


### 3.2 `get_vectors(use_rep="X_pca")` — 获取全量降维向量矩阵

```python
vectors = loader.get_vectors("X_pca")   # 使用 PCA 降维结果（推荐）
vectors = loader.get_vectors("X_umap")  # 使用 UMAP 坐标
```

**返回值：**

```python
vectors.shape  # (69032, 30) 或 (69032, 2)
vectors.dtype  # float32
```

- 返回 2D NumPy 数组，可直接传入 `ANNIndexer.build_index(vectors)`
- `use_rep` 可选值：`"X_pca"`、`"X_umap"`、`"X_tsne"` 等 `obsm` 中存在的键
- 为避免将数 GB 的表达矩阵常驻内存，当前懒加载模式下不建议用 `get_vectors("X")` 获取完整原始表达矩阵；如需读取表达矩阵，请使用 `get_X_block(start, end)` 分块读取
- 传入不存在的 `use_rep` 会抛出 `KeyError`


### 3.3 `get_vector(cell_index, use_rep="X_pca")` — 获取单个细胞向量

```python
query_vector = loader.get_vector(500, use_rep="X_pca")
```

**返回值：**

```python
query_vector.shape  # (30,)
query_vector.dtype  # float32
```

- 返回 1D NumPy 数组，可直接传入 `ANNIndexer.search(query_vector, k)`
- 对降维矩阵直接按行切片，不会展开完整表达矩阵，对大数据集性能友好
- `cell_index` 越界会抛出 `IndexError`
- `cell_index` 类型不是整数会抛出 `TypeError`


### 3.4 `get_cell_info(cell_index)` — 获取细胞元数据

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


### 3.5 `vector_dim(use_rep=None)` — 获取向量维度

```python
dim = loader.vector_dim()           # 33694（使用 X）
dim = loader.vector_dim("X_pca")    # 30（使用 X_pca）
```

- 直接用于初始化 `ANNIndexer(dim=loader.vector_dim("X_pca"))`，无需手动计算
- `vector_dim()` 不会读取完整 `X`，只返回表达矩阵特征数；真正读取表达值请使用 `get_X_block()`


### 3.6 便捷属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `loader.n_cells` | `int` | 细胞总数（69032） |
| `loader.n_genes` | `int` | 基因数量（33694） |
| `loader.obs_columns` | `list[str]` | 所有元数据字段名 |
| `loader.var_names` | `list[str]` | 基因/特征 ID（来自 `adata.var_names`） |
| `loader.var_columns` | `list[str]` | `adata.var` 中可用的基因注释列 |
| `loader.available_reps` | `list[str]` | obsm 中可用的降维键名 |


### 3.7 `cell_index_from_id(cell_id)` — 细胞 ID 反查索引

```python
idx = loader.cell_index_from_id("AAACCTGAGCAGGTCA-1_2")
# 返回整数索引，如 1234
```

- 根据细胞 ID 字符串（`obs_names`）反查其在矩阵中的整数行号
- 首次调用时懒加载构建 ID → 索引映射字典，后续查询为 O(1)
- 细胞 ID 不存在会抛出 `KeyError`

### 3.8 `get_X_block(start, end)` — 分块读取原始表达矩阵

```python
block = loader.get_X_block(0, 128)
# block.shape == (128, loader.n_genes)
# block.dtype == float32
```

- 从 `.h5ad` 的 backed 文件句柄中按行读取原始表达矩阵 `X`，不会把完整矩阵常驻内存
- 主要供 ChromaDB 初始化时计算每个细胞的 `top_genes` 使用
- `start/end` 必须满足 `0 <= start < end <= n_cells`，否则抛出 `IndexError`

### 3.9 `get_gene_names(preferred_columns=None)` — 获取可展示基因名

```python
genes = loader.get_gene_names()
```

- 默认按 `feature_name`、`gene_symbol`、`gene_name`、`symbol`、`name` 的顺序选择可读基因符号
- 若没有可用注释列，则回退使用 `var_names`
- 返回长度与 `n_genes` 一致，可与 `get_X_block()` 的列顺序一一对应

### 3.10 内部加载机制

`DataLoader` 使用 `_LightAnnDataProxy` 轻量代理代替完整 AnnData 对象。加载策略：

1. 使用 `read_h5ad(path, backed="r")` 延迟读取 HDF5 文件
2. 仅将 `obsm`（降维向量，通常 < 100MB）、`obs`（元数据，几 MB）和轻量 `var` 注释加载到内存
3. 不展开完整基因表达矩阵 `X`（可能数 GB），大幅降低内存占用
4. 读取完毕后关闭 HDF5 文件句柄

当需要访问原始表达矩阵时，可通过 `loader.get_X_block(start, end)` 分块读取，避免全量加载。


---

## 4. DatasetManager 接口说明

`DatasetManager` 管理多个 `.h5ad` 数据集，维护如下目录结构：

```text
data/
    <dataset_id>.h5ad        ← 数据文件（以 MD5 哈希命名）
    .meta/
        <dataset_id>.json    ← 元信息缓存（名称、注册时间等）
indexes/
    <dataset_id>/
        cell_index.index     ← 该数据集的 ANN 索引（由算法模块写入）
```

### 4.1 初始化

```python
from backend.data_reader import DatasetManager

manager = DatasetManager(
    data_dir="data",       # 数据文件目录
    index_dir="indexes",   # 索引文件目录
    meta_dir="data/.meta", # 元信息目录
)
```

目录不存在时自动创建。


### 4.2 `register(source_path, name=None)` — 注册已有数据集

```python
dataset_id = manager.register("data/liver.h5ad", name="Liver Atlas")
```

- 自动校验文件格式和内容有效性
- 将文件复制到 `data_dir` 并分配唯一 `dataset_id`（16位 MD5）
- 若文件已在 `data_dir` 内，可传 `copy=False` 原地注册


### 4.3 `upload(file_bytes, filename, name=None)` — 上传数据集

```python
# 在 Flask 接口中调用
dataset_id = manager.upload(
    file_bytes=request.files["file"].read(),
    filename=request.files["file"].filename,
    name=request.form.get("name"),
)
```

- 接收字节流，适合 Flask 文件上传接口
- **双重校验**：先校验文件后缀，再用 `scanpy` 读取确认内容有效
- 校验失败抛出 `ValueError`，临时文件自动清理


### 4.4 `delete_dataset(dataset_id)` — 删除数据集

```python
manager.delete_dataset(dataset_id)
```

一次性清除：

1. 内存中的 `DataLoader` 缓存
2. 数据文件（`.h5ad`）
3. 索引目录（`indexes/<dataset_id>/`）
4. 元信息文件（`.meta/<dataset_id>.json`）

- 数据集不存在会抛出 `KeyError`


### 4.5 `get_loader(dataset_id)` — 获取 DataLoader

```python
loader = manager.get_loader(dataset_id)
vectors = loader.get_vectors("X_pca")
```

- 带内存缓存，同一数据集不重复读文件
- 数据集不存在会抛出 `KeyError`


### 4.6 `list_datasets()` — 列出所有数据集

```python
datasets = manager.list_datasets()
# [
#   {
#     "dataset_id": "a3f1c2d4e5b6...",
#     "name": "Liver Atlas",
#     "filename": "a3f1c2d4e5b6....h5ad",
#     "registered_at": "2026-05-28T10:00:00",
#     "file_size_bytes": 123456789,
#   },
#   ...
# ]
```


### 4.7 `get_meta(dataset_id)` / `update_meta(dataset_id, **kwargs)` — 元信息读写

```python
meta = manager.get_meta(dataset_id)

manager.update_meta(dataset_id, name="新名称", description="儿童肝脏图谱")
```

- `dataset_id`、`filename`、`registered_at` 为不可修改字段


### 4.8 `index_path_for(dataset_id)` — 获取索引路径（供算法模块使用）

```python
index_path = manager.index_path_for(dataset_id)
# Path("indexes/a3f1c2d4e5b6.../cell_index.index")

indexer.save_index(index_path)
indexer.load_index(index_path)
```

- 目录不存在时自动创建


---

## 5. 在 `app.py` 中的调用示例

### 单数据集模式（原有方式，不变）

```python
from backend.data_reader import DataLoader
from backend.ann_indexer import ANNIndexer

USE_REP = "X_pca"
loader = DataLoader("data/liver.h5ad")
vectors = loader.get_vectors(USE_REP)
indexer = ANNIndexer(dim=loader.vector_dim(USE_REP))
indexer.build_index(vectors)

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
    return {"query_cell": cell_index, "k": k, "results": results}
```

### 多数据集模式（新增功能）

```python
from backend.data_reader import DatasetManager
from backend.ann_indexer import ANNIndexer

manager = DatasetManager()

# 启动时注册默认数据集
dataset_id = manager.register("data/liver.h5ad", name="Liver Atlas")

# 上传新数据集（Flask 接口）
@app.post("/api/datasets/upload")
def upload():
    f = request.files["file"]
    did = manager.upload(f.read(), f.filename, name=request.form.get("name"))
    return jsonify({"dataset_id": did})

# 列出所有数据集
@app.get("/api/datasets")
def list_datasets():
    return jsonify(manager.list_datasets())

# 删除数据集
@app.delete("/api/datasets/<dataset_id>")
def delete_dataset(dataset_id):
    manager.delete_dataset(dataset_id)
    return jsonify({"ok": True})

# 检索（指定数据集）
@app.get("/api/search")
def search():
    dataset_id = request.args["dataset_id"]
    cell_index = int(request.args["cell_index"])
    k = int(request.args.get("k", 10))

    loader = manager.get_loader(dataset_id)
    index_path = manager.index_path_for(dataset_id)

    indexer = ANNIndexer(dim=loader.vector_dim("X_pca"))
    if index_path.exists():
        indexer.load_index(index_path)
    else:
        indexer.build_index(loader.get_vectors("X_pca"))
        indexer.save_index(index_path)

    query_vector = loader.get_vector(cell_index, use_rep="X_pca")
    distances, indices = indexer.search(query_vector, k)
    return jsonify({"results": indices.tolist()})
```


---

## 6. 异常汇总

### DataLoader

| 异常类型 | 触发场景 |
|---|---|
| `FileNotFoundError` | 数据文件路径不存在 |
| `ValueError` | 文件后缀不是 `.h5ad` |
| `KeyError` | `use_rep` 在 `obsm` 中不存在 |
| `IndexError` | `cell_index` 超出 `[0, n_cells-1]` |
| `TypeError` | `cell_index` 不是整数类型 |

### DatasetManager

| 异常类型 | 触发场景 |
|---|---|
| `FileNotFoundError` | 注册时源文件不存在 |
| `ValueError` | 上传文件后缀不是 `.h5ad`，或内容校验失败，或文件为空 |
| `KeyError` | `dataset_id` 对应数据集不存在 |
| `ValueError` | 尝试修改不可变元信息字段 |
