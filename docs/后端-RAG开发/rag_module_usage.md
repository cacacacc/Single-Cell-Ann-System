# RAG 大模型问答模块使用说明

## 1. 模块文件

```text
backend/vector_store.py    # ChromaDB 向量数据库封装
backend/prompt_builder.py  # 细胞特征 Prompt 工程
backend/llm_client.py      # 大模型 API 接入层
backend/rag_engine.py      # RAG 完整流程引擎
.env.example               # 环境变量配置模板
```

---

## 2. 模块职责与架构

### 2.1 整体数据流

```
用户在前端输入自然语言问题
        │
        ▼  POST /api/chat
        │
        ├─ [可选] 指定 cell_index / cell_id → 使用该细胞向量作为查询
        │
        └─ [默认] 调用 LLM Embedding API 将问题文本转为向量
                │
                ▼
        ChromaDB 向量检索（vector_store.py）
        返回 Top-K 最相似细胞（含细胞类型、高表达基因、元数据）
                │
                ▼
        Prompt 组装（prompt_builder.py）
        将检索结果格式化为结构化文本上下文
        拼入 System Prompt + 用户问题 → messages 列表
                │
                ▼
        LLM API 调用（llm_client.py）
        支持智谱 GLM / OpenAI 兼容接口
                │
                ▼
        返回 JSON 给前端
        { answer, retrieved_cells, elapsed_ms, ... }
```

### 2.2 各模块职责

| 文件 | 职责 |
|---|---|
| `vector_store.py` | ChromaDB 客户端封装；将 DataLoader 中的细胞向量批量写入；提供向量相似检索接口 |
| `prompt_builder.py` | 将检索到的细胞数据（类型/基因/元数据）格式化为可读文本；自动统计检索结果细胞类型分布；组装 OpenAI-compatible messages 列表 |
| `llm_client.py` | 统一封装智谱 GLM SDK 和 OpenAI SDK；通过环境变量切换厂商；提供 `chat()`（阻塞）、`stream_chat()`（流式）和 `embed()` 接口 |
| `rag_engine.py` | 串联以上三个模块；管理多轮对话历史（session_id）；对外暴露 `ask()` 单一接口 |

### 2.3 与其他后端模块的依赖关系

```
backend/data_reader.py (DataLoader)
        │
        │  populate_from_loader(loader) 调用：
        │    loader.get_vectors()     ← 获取全量 PCA 向量矩阵
        │    loader.get_vector()      ← 获取单细胞原始表达向量（计算高表达基因用）
        │    loader.get_cell_info()   ← 获取细胞元数据（cell_type、tissue 等）
        │    loader.obs_columns       ← 获取 obs 字段列表
        │    loader.adata.var_names   ← 获取基因名称列表
        ▼
backend/vector_store.py (CellVectorStore)
        │
        │  query_similar() 被调用：
        │    ← backend/rag_engine.py (RAGEngine.ask)
        │    ← app.py (/api/vectordb/query)
        │    ← app.py (/api/chat)
        │    ← app.py (/api/chat/stream)
        ▼
ChromaDB 持久化文件（chroma_db/ 目录）
```

**与 `ann_indexer.py` 的关系**：两套检索路径并行独立，互不干扰：
- ANN 索引（faiss/hnswlib）→ `/api/search`（原有功能）
- ChromaDB 向量数据库 → `/api/chat`、`/api/chat/stream`、`/api/vectordb/query`（本模块）

---

## 3. 环境配置

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

与本模块相关的依赖：

```text
python-dotenv>=1.0    # .env 文件自动加载（Windows/Mac/Linux 通用）
chromadb>=0.5.0       # 向量数据库
zhipuai>=2.1.0        # 智谱 GLM SDK（推荐）
openai>=1.0.0         # OpenAI 兼容 SDK（可选）
```

> `zhipuai` 和 `openai` 二选一安装即可，两个都装也没问题。

### 3.2 配置 API Key

**第一步**：复制配置模板

```bash
# Windows（命令提示符）
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

**第二步**：用任意文本编辑器打开 `.env`，填入 API Key

```ini
# 推荐：智谱 GLM（国内访问稳定，注册即有免费额度）
# 申请地址：https://open.bigmodel.cn/
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=你的Key.填在这里
LLM_MODEL=glm-4-flash
LLM_EMBEDDING_MODEL=embedding-3
```

> `.env` 文件已在 `.gitignore` 中，不会被提交到 git，每个人填自己的 Key。

**可选方案（任意 OpenAI 兼容接口）**：

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1   # DeepSeek 示例
LLM_MODEL=deepseek-chat
```

支持 DeepSeek、阿里通义千问、月之暗面 Kimi 等兼容 OpenAI 协议的服务。

---

## 4. 对外 API 接口

### 4.1 接口总览

| 方法 | 路径 | 功能 | 权限 |
|---|---|---|---|
| POST | `/api/vectordb/init` | 将数据集细胞向量写入 ChromaDB | 登录用户 |
| GET | `/api/vectordb/status` | 查询向量库状态（数量/是否就绪/use_rep） | 登录用户 |
| POST | `/api/vectordb/query` | 直接向量检索（不经 LLM） | 登录用户 |
| DELETE | `/api/vectordb/collection` | 清空向量库 | 管理员 |
| **POST** | **`/api/chat`** | **RAG 问答核心接口（阻塞，返回完整 JSON）** | 登录用户 |
| **POST** | **`/api/chat/stream`** | **RAG 问答流式接口（SSE，逐字返回）** | 登录用户 |
| GET | `/api/chat/history` | 查询会话对话历史 | 登录用户 |
| DELETE | `/api/chat/history` | 清空会话对话历史 | 登录用户 |
| GET | `/api/chat/sessions` | 列出所有对话会话（SQLite + 内存合并） | 登录用户 |
| GET | `/api/chat/sessions/<id>` | 获取指定对话的全部消息 | 登录用户 |
| DELETE | `/api/chat/sessions/<id>` | 删除指定对话（消息一并删除） | 登录用户 |
| PATCH | `/api/chat/sessions/<id>` | 重命名对话标题 | 登录用户 |
| GET | `/api/llm/info` | 查看 LLM 配置信息 | 登录用户 |
| POST | `/api/llm/ping` | 测试 LLM API 连通性 | 登录用户 |

### 4.2 初始化向量数据库

**POST /api/vectordb/init**

```bash
curl -X POST http://localhost:5000/api/vectordb/init \
  -H "Content-Type: application/json" \
  -d '{"use_rep": "X_pca", "distance_metric": "cosine"}'
```

首次使用或更换数据集后调用，将细胞数据写入 ChromaDB。初始化完成后数据持久化在 `chroma_db/` 目录，**重启服务不需要重新初始化**。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `use_rep` | string | `"X_pca"` | 使用的降维表示 |
| `distance_metric` | string | `"cosine"` | 距离函数：`cosine` / `l2` / `ip` |
| `force` | bool | `false` | `true` 时先清空再重写 |
| `top_genes` | int | `20` | 每个细胞记录前 N 个高表达基因 |

### 4.3 RAG 问答核心接口

**POST /api/chat**

**请求**

```json
{
  "question": "这些细胞主要执行哪些生物学功能？",
  "cell_index": 42,
  "use_rep": "X_pca",
  "n_results": 5,
  "session_id": "user_123_session_1"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `question` | string | ✅ | 用户的自然语言问题 |
| `dataset_id` | string | — | 数据集 ID，省略则用默认数据集 |
| `cell_index` | int | — | 用该细胞向量做检索（与 cell_id 二选一） |
| `cell_id` | string | — | 用该细胞向量做检索（与 cell_index 二选一） |
| `use_rep` | string | — | 向量表示，默认 `X_pca` |
| `n_results` | int | — | 检索细胞数，默认 `5` |
| `session_id` | string | — | 会话 ID，提供时保留多轮对话历史 |
| `where` | object | — | ChromaDB 元数据过滤，如 `{"cell_type": {"$eq": "Hepatocyte"}}` |
| `preset` | string | — | LLM 角色预设：`bioinfo_expert`（生信分析专家）、`strict_analyst`（严谨数据统计员）、`science_communicator`（通俗科普助手） |
| `temperature` | float | — | 覆盖默认生成温度（0~1） |
| `max_tokens` | int | — | 覆盖默认最大生成 token 数 |

> `cell_index` / `cell_id` 与 `question` 的关系：
> - 提供 `cell_index` 或 `cell_id`：用该细胞的向量做相似检索，问题直接喂给 LLM
> - 都不提供：系统调用 Embedding API 把问题文本转向量后检索

**响应**

```json
{
  "answer": "这批细胞高表达 ALB、APOA1、CYP3A4 等基因，为肝细胞（Hepatocyte），主要功能是...",
  "retrieved_cells": [
    {
      "rank": 1,
      "cell_id": "ACGT-1",
      "cell_type": "Hepatocyte",
      "distance": 0.08,
      "top_genes": "ALB,APOA1,CYP3A4,FABP1,...",
      "document": "Cell type: Hepatocyte | Top genes: ALB...",
      "metadata": { "tissue": "liver", "leiden": "2" }
    }
  ],
  "context_used": "## 检索到的相似细胞数据...",
  "query_vectorized": false,
  "elapsed_ms": 1823.4,
  "retrieve_ms": 12.1,
  "llm_ms": 1790.2,
  "session_id": "user_123_session_1",
  "model": "glm-4-flash",
  "n_retrieved": 5
}
```

### 4.4 RAG 问答流式接口（SSE）

**POST /api/chat/stream**

与 `/api/chat` 请求参数完全相同，区别在于响应格式为 `text/event-stream`，大模型回答**逐字实时推送**，前端可实现打字机效果。

**响应格式（SSE）**

```
data: 这批细胞高表达
data:  ALB、APOA1
data: ，为肝细胞...
data: [FORMATTED] ## 分析结果\n\n这批细胞高表达 **ALB**...
data: [SOURCES] [{"rank":1,"cell_id":"ACGT-1","cell_type":"Hepatocyte",...}]
data: [DONE]
```

- 每个 `data:` 事件包含一个文本片段
- `[FORMATTED]` 事件：流结束后发送 Markdown 格式化后的完整文本（含基因名加粗、标题分段等增强）
- `[SOURCES]` 事件：发送检索到的相似细胞 JSON 数组，前端可渲染为溯源卡片
- 最后一个事件固定为 `data: [DONE]`，表示流结束
- 若中途出错，发送 `data: [ERROR] 错误信息`

**前端接入示例（fetch + ReadableStream）**

```javascript
const resp = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: '这些细胞有什么功能？', cell_index: 0 }),
});
const reader = resp.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of decoder.decode(value).split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const chunk = line.slice(6);
    if (chunk === '[DONE]') return;
    document.getElementById('answer').innerText += chunk;
  }
}
```

> 流式接口同样支持 `session_id` 多轮对话，回答完成后自动写入历史记录。

---

### 4.5 直接向量检索（不经 LLM）

**POST /api/vectordb/query**

```json
{
  "cell_index": 42,
  "n_results": 10,
  "where": { "cell_type": { "$eq": "Hepatocyte" } }
}
```

返回与指定细胞最相似的 Top-K 个细胞，结构同 `retrieved_cells` 数组。

### 4.6 对话会话管理

```bash
# 列出所有对话会话（SQLite + 内存合并，按最近更新倒序）
GET /api/chat/sessions
# 返回：{ "sessions": [{ "id": "xxx", "title": "肝细胞功能", "message_count": 6, ... }] }

# 获取指定对话的全部消息
GET /api/chat/sessions/<session_id>
# 返回：{ "session_id": "xxx", "messages": [{ "role": "user", "content": "...", "created_at": "..." }] }

# 删除指定对话（消息一并删除）
DELETE /api/chat/sessions/<session_id>

# 重命名对话标题
curl -X PATCH http://localhost:5000/api/chat/sessions/<session_id> \
  -H "Content-Type: application/json" \
  -d '{"title": "肝细胞基因表达分析"}'

# 查询内存对话历史（回退到内存 ChatHistory）
GET /api/chat/history?session_id=xxx

# 清空内存对话历史
DELETE /api/chat/history?session_id=xxx
```

### 4.7 LLM 辅助接口

```bash
# 查看当前 LLM 配置（不含 API Key 明文）
GET /api/llm/info
# 返回：{ "provider": "zhipu", "model": "glm-4-flash", "is_available": true, "api_key_preview": "xxx...xxx" }

# 测试 LLM API 连通性
POST /api/llm/ping
# 返回：{ "ok": true, "model": "glm-4-flash", "reply": "OK", "elapsed_ms": 430 }
```

---

## 5. 向量数据库模块说明（vector_store.py）

### 5.1 写入数据结构

写入 ChromaDB 时，每个细胞对应一条记录：

| ChromaDB 字段 | 内容 |
|---|---|
| `id` | 细胞的字符串 ID（来自 `obs_names`） |
| `embedding` | PCA 向量（float 列表，默认 50 维） |
| `metadata` | `cell_index`、`cell_id`、`cell_type`、`tissue`、`leiden` 等所有 obs 字段，以及 `top_genes`（高表达基因逗号分隔） |
| `document` | 可读文本：`Cell type: X \| Tissue: Y \| Top genes: ALB, APOA1, ...` |

### 5.2 Python 代码直接调用

```python
from backend.vector_store import CellVectorStore, get_or_create_store
from backend.data_reader import DataLoader

# 写入数据
loader = DataLoader("data/liver.h5ad")
store = CellVectorStore(collection_name="liver", persist_dir="chroma_db")
count = store.populate_from_loader(loader, use_rep="X_pca", top_genes=20)

# 向量相似检索
results = store.query_similar(query_vector, n_results=5)

# 带元数据过滤的检索
results = store.query_similar(
    query_vector=query_vec,
    n_results=5,
    where={"cell_type": {"$eq": "Hepatocyte"}},
)

# 纯元数据过滤（不做向量检索）
results = store.query_by_metadata(
    where={"cell_type": {"$eq": "T cell"}},
    limit=20,
)
```

### 5.3 注意事项

- **向量维度必须一致**：查询向量的维度须与写入时使用的 `use_rep` 维度相同（`X_pca` 默认 50 维）。
- **`distance_metric` 一旦写入不可更改**：需更换时须先调用 `DELETE /api/vectordb/collection` 清空后重新初始化。
- **持久化目录**：数据在 `chroma_db/` 目录，重启服务无需重新写入；手动删除该目录后需重新初始化。
- **`force=True` 的代价**：先删除整个 Collection 再重建，肝脏数据集（~5000 细胞）写入约 30 秒。

---

## 6. 验证功能是否正常

```bash
# 步骤一：验证 LLM API Key 是否有效
curl -X POST http://localhost:5000/api/llm/ping
# 返回 "ok": true 表示配置正确

# 步骤二：验证向量数据库是否就绪
curl http://localhost:5000/api/vectordb/status
# 返回 "is_populated": true 表示数据已写入，可以开始问答
# 若为 false，先调用 POST /api/vectordb/init 初始化

# 步骤三：发起一次问答测试
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "数据集中有哪些细胞类型？", "cell_index": 0}'
```

---

## 7. 常见问题

**Q：`向量数据库尚未初始化` 报错**  
A：先调用 `POST /api/vectordb/init` 写入数据，只需做一次。

**Q：`大模型调用失败` 报错**  
A：检查 `.env` 中的 API Key 是否正确，用 `POST /api/llm/ping` 验证连通性。

**Q：没有 `cell_index` 时报 `Embedding API 调用失败`**  
A：纯文本问答需要 Embedding API。智谱 GLM 的 `embedding-3` 支持此功能，确认 `.env` 中 `LLM_EMBEDDING_MODEL=embedding-3` 已设置。或者在请求中带上 `cell_index` 参数，用细胞向量代替文本向量。

**Q：`chromadb 未安装` 报错**  
A：执行 `pip install chromadb` 或重新执行 `pip install -r requirements.txt`。

**Q：多轮对话历史是否持久化？**  
A：已支持双存储。内存中通过 `ChatHistory` 缓存（最多 5 轮，TTL 1 小时），同时持久化到 SQLite 数据库（`chat_sessions` 和 `chat_messages` 表）。服务重启后 SQLite 数据仍在。通过 `GET /api/chat/sessions` 可列出所有历史会话，通过 `GET /api/chat/sessions/<id>` 可获取完整消息。

**Q：如何使用关键词检索回退？**  
A：当请求中不带 `cell_index` / `cell_id` 时，系统会先尝试调用 Embedding API 将问题文本转为向量。如果 Embedding 调用失败或 LLM 未配置 Embedding 模型，系统会自动从问题中提取基因名和细胞类型关键词，通过 ChromaDB 文档字段进行关键词匹配检索。如果关键词也无匹配结果，则回退到全库随机采样。
