# Web 接口模块使用说明

## 1. 模块文件

```text
app.py
```

当前项目以根目录 `app.py` 作为 Flask Web 服务主入口，负责连接前端页面、数据读取模块 `DataLoader` 和 ANN 检索模块 `ANNIndexer`。

## 2. 主要功能

- 启动 Flask 服务并挂载前端模板页面
- 开启 CORS，允许前端跨域访问
- 支持上传、列出、删除 `.h5ad` 数据集
- 查询数据集元信息、细胞列表和 UMAP 坐标
- 按参数构建或加载 ANN 索引
- 执行 Top-K 相似细胞检索并返回 JSON 结果
- 执行 ANN 算法性能评测并保存历史记录

## 3. 运行环境

建议使用项目虚拟环境安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

默认数据文件路径为：

```text
data/liver.h5ad
```

## 4. 启动服务

从项目根目录启动：

```bash
source .venv/bin/activate
PORT=5001 python app.py
```

启动后访问：

```text
http://127.0.0.1:5001
```

## 5. 索引参数配置

索引参数既可以通过环境变量设置，也可以在 `/api/metadata`、`/api/index/build`、`/api/search` 请求参数中传入。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `index_backend` / `CELL_INDEX_BACKEND` | `auto` | 索引后端：`auto` / `faiss` / `hnswlib` / `numpy` |
| `index_type` / `CELL_INDEX_TYPE` | `flat` | 索引类型：`flat` / `ivf_flat` / `hnsw` / `pq` / `brute` |
| `index_metric` / `CELL_INDEX_METRIC` | `l2` | 距离度量：`l2` / `cosine` / `ip` |
| `nlist` / `CELL_INDEX_NLIST` | `100` | IVF 分桶数量 |
| `nprobe` / `CELL_INDEX_NPROBE` | `10` | IVF 查询探测数量 |
| `m` / `CELL_INDEX_M` | `16` | HNSW 图的 M 参数 |
| `ef_construction` / `CELL_INDEX_EF_CONSTRUCTION` | `200` | HNSW 构建参数 |
| `ef_search` / `CELL_INDEX_EF_SEARCH` | `50` | HNSW 查询参数 |
| `pq_m` / `CELL_INDEX_PQ_M` | `8` | PQ 子空间数量 |
| `pq_nbits` / `CELL_INDEX_PQ_NBITS` | `8` | PQ 编码位数 |

## 6. API 接口

### 6.1 健康检查

```http
GET /api/health
```

返回默认数据集、默认向量表示和默认索引状态。`ready=false` 表示默认索引尚未构建，但服务仍可继续使用上传、构建索引等接口。

### 6.2 数据集元信息

```http
GET /api/metadata?dataset_id=liver&use_rep=X_pca
```

返回数据集规模、可用向量表示、obs 元数据列名、PQ 参数建议和当前索引配置。

### 6.3 数据集管理

```http
GET /api/datasets
POST /api/datasets/upload
DELETE /api/datasets/<dataset_id>
```

`POST /api/datasets/upload` 使用 multipart form 上传 `.h5ad` 文件，字段名为 `file`。

### 6.4 细胞分页列表

```http
GET /api/cells?dataset_id=liver&offset=0&limit=50
```

返回细胞 ID、内部编号、细胞类型和分页信息。`limit` 最大为 `200`。

### 6.5 UMAP 坐标

```http
GET /api/umap?dataset_id=liver&limit=3000&seed=42&color_by=cell_type
POST /api/umap/cells
```

`GET /api/umap` 用于前端散点图抽样展示。`POST /api/umap/cells` 根据一组 `cell_ids` 返回指定细胞的 UMAP 坐标，最多 `500` 个。

### 6.6 构建索引

```http
POST /api/index/build
Content-Type: application/json

{
  "dataset_id": "liver",
  "use_rep": "X_pca",
  "index_backend": "faiss",
  "index_type": "hnsw",
  "index_metric": "l2",
  "m": 16,
  "ef_construction": 200,
  "ef_search": 50
}
```

返回构建后的后端、索引类型、距离度量和完整索引配置。若使用 `pq`，服务会根据向量维度自动修正不合法的 `pq_m`。

### 6.7 相似细胞检索

```http
POST /api/search
Content-Type: application/json

{
  "dataset_id": "liver",
  "cell_id": "AAACCTGAGCAGGTCA-1_2",
  "k": 10,
  "include_self": false,
  "use_rep": "X_pca",
  "index_backend": "auto",
  "index_type": "flat",
  "index_metric": "l2"
}
```

也可以使用 `cell_index` 查询：

```http
GET /api/search?dataset_id=liver&cell_index=500&k=10
```

当前版本的 `/api/search` 会在索引缺失时按请求参数自动构建索引，因此第一次检索可能耗时较长。前端若希望提前展示进度，可先调用 `/api/index/build`。

返回示例：

```json
{
  "dataset_id": "liver",
  "query_cell": 500,
  "cell_id": "AAACCTGAGCAGGTCA-1_2",
  "k": 10,
  "include_self": false,
  "use_rep": "X_pca",
  "index_backend": "faiss",
  "index_type": "flat",
  "index_metric": "l2",
  "elapsed_ms": 1.27,
  "index_prepare_ms": 0.18,
  "total_elapsed_ms": 2.01,
  "results": [
    {
      "rank": 1,
      "cell_index": 1024,
      "cell_id": "AAACCTGAGCAGGTCA-1_3",
      "cell_type": "hepatocyte",
      "distance": 0.235419,
      "similarity_score": 0.809442,
      "metadata": {
        "cell_id": "AAACCTGAGCAGGTCA-1_3",
        "cell_type": "hepatocyte"
      }
    }
  ]
}
```

### 6.8 性能评测

```http
POST /api/benchmark
GET /api/benchmark/history
```

`POST /api/benchmark` 会比较暴力精确检索、FAISS-HNSW、HNSWLIB、FAISS-IVF、FAISS-PQ 的平均耗时和 recall 曲线，并将结果写入 `data/benchmark_history.json`。

### 6.9 数据集索引列表

```http
GET /api/datasets/<dataset_id>/indices
```

返回指定数据集的所有已构建索引，包括每个索引的名称、配置信息和就绪状态。

```json
{
  "indices": [
    {
      "name": "X_pca_flat_l2",
      "config": {
        "backend": "faiss",
        "index_type": "flat",
        "metric": "l2"
      },
      "ready": true
    }
  ],
  "ready": true
}
```

### 6.10 删除索引

```http
DELETE /api/index/<dataset_id>/<index_name>
```

删除指定数据集的单个命名索引。需要管理员权限。

### 6.11 搜索快照

```http
GET /api/profile/search-snapshots?limit=20
DELETE /api/profile/search-snapshots/<snapshot_id>
```

每次执行 `/api/search` 时会自动保存搜索快照（含参数、耗时、结果数），用户可在个人中心查看和一键重跑。`GET` 返回快照列表，`DELETE` 删除指定快照。

### 6.12 向量数据库（ChromaDB）

```http
POST /api/vectordb/init
GET /api/vectordb/status
POST /api/vectordb/query
DELETE /api/vectordb/collection
```

`POST /api/vectordb/init` 将数据集细胞向量写入 ChromaDB，首次使用 AI 助手前需调用。初始化后数据持久化在 `chroma_db/` 目录，重启服务不需要重新初始化。

`GET /api/vectordb/status` 返回 Collection 状态（数量、是否就绪、use_rep）。

`POST /api/vectordb/query` 直接执行向量检索，不经 LLM。

`DELETE /api/vectordb/collection` 清空向量库，需要管理员权限。

### 6.13 RAG AI 问答

```http
POST /api/chat
POST /api/chat/stream
```

`/api/chat` 为阻塞式 RAG 问答，返回完整 JSON 结果（answer、retrieved_cells、耗时等）。

`/api/chat/stream` 为 SSE 流式问答，逐字返回大模型回答。流结束后发送 `[FORMATTED]`（Markdown 格式化全文）、`[SOURCES]`（检索来源）和 `[DONE]` 事件。

两个接口均支持以下可选参数：`preset`（角色预设）、`temperature`、`max_tokens`、`session_id`（多轮对话）。

### 6.14 对话会话管理

```http
GET /api/chat/sessions
GET /api/chat/sessions/<session_id>
DELETE /api/chat/sessions/<session_id>
PATCH /api/chat/sessions/<session_id>
GET /api/chat/history?session_id=xxx
DELETE /api/chat/history?session_id=xxx
```

`GET /api/chat/sessions` 列出当前用户的所有对话会话（SQLite 持久化 + 内存会话合并）。

`GET /api/chat/sessions/<id>` 获取指定对话的全部消息。

`DELETE /api/chat/sessions/<id>` 删除指定对话（消息一并删除）。

`PATCH /api/chat/sessions/<id>` 重命名对话标题（请求体 `{"title": "新标题"}`）。

`GET /api/chat/history` 和 `DELETE /api/chat/history` 用于查询和清空内存对话历史。

### 6.15 LLM 辅助接口

```http
GET /api/llm/info
POST /api/llm/ping
```

`/api/llm/info` 返回当前 LLM 配置信息（不含 API Key 明文）。

`/api/llm/ping` 向大模型发送测试请求，验证 API Key 和网络连通性。

## 7. 常见异常

| 状态码 | 场景 |
|---|---|
| `400` | 参数缺失、参数类型错误、`k <= 0`、`k > 100`、细胞编号越界、索引参数不合法 |
| `404` | 搜索时指定的数据集不存在 |
| `409` | 上传数据集时文件已存在 |
| `503` | 默认数据集不存在、索引运行时不可用或服务暂不可用 |

## 8. 与其他后端模块的关系

```text
前端页面
   |
   v
app.py
   |
   +-- backend/data_reader.py       读取 .h5ad 数据和细胞元数据
   |
   +-- backend/ann_indexer.py       构建 ANN 索引并执行 Top-K 检索
   |
   +-- backend/vector_store.py      ChromaDB 向量数据库（AI 问答检索路径）
   |
   +-- backend/rag_engine.py        RAG 流程编排（检索 + Prompt + LLM）
   |
   +-- backend/llm_client.py        大模型 API 调用（智谱 GLM / OpenAI）
   |
   +-- backend/prompt_builder.py    Prompt 工程与上下文组装
   |
   +-- backend/user_store.py        用户认证、快照和对话消息持久化
```
