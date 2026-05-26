# Web 接口模块使用说明

## 1. 模块文件

```text
backend/app.py
```

该模块负责把数据读取模块 `DataLoader` 和 ANN 检索模块 `ANNIndexer` 封装成 Flask Web 服务，供前端通过 HTTP 接口调用。

## 2. 主要功能

- 启动 Flask 服务
- 开启 CORS，允许前端跨域访问
- 服务启动时加载 `.h5ad` 数据文件
- 使用指定向量表示构建或加载 ANN 索引
- 提供健康检查、元数据查询和相似细胞检索接口
- 将检索结果整理成 JSON 返回给前端

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

如果数据文件不在默认位置，可以通过环境变量指定：

```bash
CELL_DATA_PATH=/path/to/liver.h5ad PORT=5001 python3 -m backend.app
```

## 4. 启动服务

```bash
source .venv/bin/activate
PORT=5001 python3 -m backend.app
```

启动后默认访问地址为：

```text
http://127.0.0.1:5001
```

如果 `5000` 端口被占用，可以使用 `PORT=5001` 或其他端口。

## 5. 环境变量配置

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `CELL_DATA_PATH` | `data/liver.h5ad` | 单细胞 `.h5ad` 数据文件路径 |
| `CELL_INDEX_PATH` | `indexes/cell_index.index` | ANN 索引保存路径 |
| `CELL_USE_REP` | `X_pca` | 使用的向量表示，如 `X`、`X_pca` |
| `PORT` | `5000` | Flask 服务端口 |

## 6. API 接口

### 6.1 健康检查

```http
GET /api/health
```

服务正常并且数据、索引加载成功时返回 `200`：

```json
{
  "ready": true,
  "data_path": "data/liver.h5ad",
  "index_path": "indexes/cell_index.index",
  "use_rep": "X_pca",
  "n_cells": 69032,
  "n_genes": 33694,
  "vector_dim": 30,
  "index_backend": "faiss"
}
```

如果数据文件不存在或初始化失败，返回 `503`：

```json
{
  "ready": false,
  "data_path": "data/liver.h5ad",
  "index_path": "indexes/cell_index.index",
  "use_rep": "X_pca",
  "error": "数据文件不存在：data/liver.h5ad"
}
```

### 6.2 数据集元信息

```http
GET /api/metadata
```

返回数据集规模、可用向量表示、obs 元数据列名和索引后端等信息。

### 6.3 相似细胞检索

```http
GET /api/search?cell_index=500&k=10
```

也支持 POST JSON：

```http
POST /api/search
Content-Type: application/json

{
  "cell_index": 500,
  "k": 10,
  "include_self": false
}
```

参数说明：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `cell_index` | int | 是 | 无 | 查询细胞的编号，从 0 开始 |
| `k` | int | 否 | `10` | 返回最相似的前 K 个细胞 |
| `include_self` | bool | 否 | `false` | 是否在结果中包含查询细胞自身 |

返回示例：

```json
{
  "query_cell": 500,
  "k": 10,
  "include_self": false,
  "use_rep": "X_pca",
  "results": [
    {
      "rank": 1,
      "cell_index": 1024,
      "cell_id": "AAACCTGAGCAGGTCA-1_2",
      "cell_type": "hepatocyte",
      "distance": 0.235419,
      "similarity_score": 0.809442,
      "metadata": {
        "cell_id": "AAACCTGAGCAGGTCA-1_2",
        "cell_type": "hepatocyte",
        "donor_id": "D1"
      }
    }
  ]
}
```

## 7. 异常情况

| 状态码 | 场景 |
|---|---|
| `400` | 参数缺失、参数类型错误、`k <= 0`、`k > 100`、细胞编号越界 |
| `503` | 服务未准备好，例如数据文件不存在、索引未成功构建 |

## 8. 前端调用示例

```javascript
async function searchSimilarCells(cellIndex, k) {
  const response = await fetch(
    `http://127.0.0.1:5001/api/search?cell_index=${cellIndex}&k=${k}`
  );
  const data = await response.json();
  return data.results;
}
```

## 9. 与其他后端模块的关系

```text
前端页面
   |
   v
backend/app.py
   |
   +-- backend/data_reader.py   读取 .h5ad 数据和细胞元数据
   |
   +-- backend/ann_indexer.py   构建 ANN 索引并执行 Top-K 检索
```

`app.py` 不直接处理底层数据格式和 ANN 算法细节，只负责接收请求、调用底层模块、整理 JSON 响应，是前端和算法模块之间的接口层。
